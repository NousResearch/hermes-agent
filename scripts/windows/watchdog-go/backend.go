package main

import (
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
)

const desktopBackendManifestName = "desktop-backend.json"

// DefaultManagedBackendPort is outside reserved ops ports (9119 dashboard serve, 9120 dashboard UI, …).
const DefaultManagedBackendPort = 9118

var backendReadyRE = regexp.MustCompile(`^HERMES_(?:BACKEND|DASHBOARD)_READY port=(\d+)`)

// DesktopBackendManifest is published for packaged Desktop to connect without cold-spawning serve.
type DesktopBackendManifest struct {
	BaseURL    string `json:"baseUrl"`
	Token      string `json:"token"`
	Port       int    `json:"port"`
	PID        int    `json:"pid,omitempty"`
	HermesRoot string `json:"hermesRoot,omitempty"`
	HermesHome string `json:"hermesHome,omitempty"`
	UpdatedAt  string `json:"updatedAt"`
	Managed    bool   `json:"managed"`
}

// BackendManager supervises a watchdog-owned hermes serve for fast Desktop connect.
type BackendManager struct {
	cfg    Config
	logger *Logger

	mu    sync.Mutex
	cmd   *exec.Cmd
	pid   int
	port  int
	token string
}

func NewBackendManager(cfg Config, logger *Logger) *BackendManager {
	return &BackendManager{cfg: cfg, logger: logger}
}

func (bm *BackendManager) ManifestPath() string {
	return filepath.Join(bm.cfg.DataDir, desktopBackendManifestName)
}

func parseReadyPortLine(line string) (int, bool) {
	m := backendReadyRE.FindStringSubmatch(strings.TrimSpace(line))
	if len(m) != 2 {
		return 0, false
	}
	var port int
	if _, err := fmt.Sscanf(m[1], "%d", &port); err != nil || port <= 0 {
		return 0, false
	}
	return port, true
}

func generateSessionToken() (string, error) {
	buf := make([]byte, 32)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(buf), nil
}

func resolvePythonExe(hermesRoot string) string {
	if hermesRoot == "" {
		return ""
	}
	for _, rel := range []string{".venv\\Scripts\\python.exe", "venv\\Scripts\\python.exe"} {
		candidate := filepath.Join(hermesRoot, rel)
		if fileExists(candidate) {
			return candidate
		}
	}
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		shared := filepath.Join(home, ".hermes", "hermes-agent", "venv", "Scripts", "python.exe")
		if fileExists(shared) {
			return shared
		}
	}
	return ""
}

func resolveWebDist(hermesRoot string) string {
	if hermesRoot == "" {
		return ""
	}
	candidate := filepath.Join(hermesRoot, "hermes_cli", "web_dist")
	if fileExists(filepath.Join(candidate, "index.html")) {
		return candidate
	}
	return candidate
}

func buildServeCommand(cfg Config) (*exec.Cmd, string, int, error) {
	python := resolvePythonExe(cfg.HermesRoot)
	if python == "" {
		return nil, "", 0, fmt.Errorf("python not found under %s (.venv or venv)", cfg.HermesRoot)
	}
	token, err := generateSessionToken()
	if err != nil {
		return nil, "", 0, err
	}
	port := cfg.ManagedBackendPort
	if port <= 0 {
		port = DefaultManagedBackendPort
	}
	if isReservedOpsPort(port) {
		return nil, "", 0, fmt.Errorf("managed backend port %d is reserved for ops services", port)
	}
	webDist := resolveWebDist(cfg.HermesRoot)
	cmd := exec.Command(
		python,
		"-m", "hermes_cli.main",
		"serve",
		"--host", "127.0.0.1",
		"--port", fmt.Sprintf("%d", port),
	)
	cmd.Dir = cfg.HermesRoot
	cmd.Env = append(os.Environ(),
		"HERMES_HOME="+cfg.HermesHome,
		"HERMES_DESKTOP=1",
		"HERMES_WATCHDOG_MANAGED=1",
		"HERMES_DASHBOARD_SESSION_TOKEN="+token,
		"HERMES_WEB_DIST="+webDist,
		"HERMES_DESKTOP_HERMES_ROOT="+cfg.HermesRoot,
		"HERMES_DESKTOP_CWD="+cfg.HermesRoot,
		"PYTHONUTF8=1",
		"PYTHONIOENCODING=utf-8",
		"PYTHONUNBUFFERED=1",
	)
	return cmd, token, port, nil
}

func (bm *BackendManager) readManifest() (*DesktopBackendManifest, error) {
	raw, err := os.ReadFile(bm.ManifestPath())
	if err != nil {
		return nil, err
	}
	var manifest DesktopBackendManifest
	if err := json.Unmarshal(raw, &manifest); err != nil {
		return nil, err
	}
	return &manifest, nil
}

func (bm *BackendManager) writeManifest(manifest DesktopBackendManifest) error {
	raw, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(bm.ManifestPath(), raw, 0o644)
}

func (bm *BackendManager) clearManifest() {
	_ = os.Remove(bm.ManifestPath())
}

func (bm *BackendManager) currentHealthy() *backendInfo {
	bm.mu.Lock()
	pid := bm.pid
	port := bm.port
	bm.mu.Unlock()
	if pid <= 0 || port <= 0 {
		return nil
	}
	if !processAlive(pid) {
		return nil
	}
	if isReservedOpsPort(port) {
		return nil
	}
	if !testBackendStatus(port) {
		return nil
	}
	return &backendInfo{PID: uint32(pid), Port: port, Cmd: "watchdog-managed serve"}
}

func (bm *BackendManager) stopLocked() {
	if bm.cmd != nil && bm.cmd.Process != nil {
		stopProcessPID(uint32(bm.cmd.Process.Pid))
	}
	bm.cmd = nil
	bm.pid = 0
	bm.port = 0
	bm.token = ""
}

func (bm *BackendManager) waitForReadyPort(port int, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if testBackendStatus(port) {
			return nil
		}
		time.Sleep(500 * time.Millisecond)
	}
	return fmt.Errorf("timed out waiting for /api/status on port %d (%s)", port, timeout)
}

// EnsureHealthy keeps (or starts) the watchdog-managed serve and publishes desktop-backend.json.
func (bm *BackendManager) EnsureHealthy() (*backendInfo, error) {
	if bm.cfg.HermesRoot == "" {
		return nil, fmt.Errorf("hermes root not configured")
	}
	if existing := bm.currentHealthy(); existing != nil {
		_ = bm.publishManifestLocked(existing.Port, int(existing.PID))
		return existing, nil
	}

	bm.mu.Lock()
	defer bm.mu.Unlock()

	if bm.cmd != nil && bm.port > 0 && processAlive(bm.pid) && testBackendStatus(bm.port) {
		info := &backendInfo{PID: uint32(bm.pid), Port: bm.port, Cmd: "watchdog-managed serve"}
		_ = bm.publishManifestLocked(bm.port, bm.pid)
		return info, nil
	}

	bm.stopLocked()

	if port := bm.cfg.ManagedBackendPort; port <= 0 {
		port = DefaultManagedBackendPort
	} else if isReservedOpsPort(port) {
		bm.clearManifest()
		return nil, fmt.Errorf("managed backend port %d is reserved", port)
	} else if testBackendStatus(port) {
		bm.port = port
		if manifest, err := bm.readManifest(); err == nil && manifest.Token != "" {
			bm.token = manifest.Token
		}
		if bm.token == "" {
			token, terr := generateSessionToken()
			if terr != nil {
				return nil, terr
			}
			bm.token = token
		}
		_ = bm.publishManifestLocked(port, 0)
		bm.logger.Infof("reusing healthy managed backend on port %d", port)
		return &backendInfo{Port: port, Cmd: "existing serve on managed port"}, nil
	}

	cmd, token, port, err := buildServeCommand(bm.cfg)
	if err != nil {
		bm.clearManifest()
		return nil, err
	}
	hideWindowsProcess(cmd)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	cmd.Stderr = io.Discard

	if err := cmd.Start(); err != nil {
		bm.clearManifest()
		return nil, err
	}
	go io.Copy(io.Discard, stdout)

	if err := bm.waitForReadyPort(port, time.Duration(bm.cfg.BackendStartTimeoutSec)*time.Second); err != nil {
		if cmd.Process != nil && !processAlive(cmd.Process.Pid) {
			bm.stopLocked()
			bm.clearManifest()
			return nil, fmt.Errorf("managed backend exited before /api/status became ready")
		}
		// Child uvicorn may outlive the parent wrapper — keep waiting on the fixed port.
		if err2 := bm.waitForReadyPort(port, time.Duration(bm.cfg.BackendReadyTimeoutSec)*time.Second); err2 != nil {
			bm.stopLocked()
			bm.clearManifest()
			return nil, err2
		}
	}

	bm.cmd = cmd
	if cmd.Process != nil {
		bm.pid = cmd.Process.Pid
	} else {
		bm.pid = 0
	}
	bm.port = port
	bm.token = token

	if err := bm.publishManifestLocked(port, bm.pid); err != nil {
		bm.logger.Infof("manifest write failed: %v", err)
	}

	bm.logger.Infof("managed backend ready pid=%d port=%d", bm.pid, bm.port)
	return &backendInfo{PID: uint32(bm.pid), Port: port, Cmd: "watchdog-managed serve"}, nil
}

func (bm *BackendManager) publishManifestLocked(port, pid int) error {
	manifest := DesktopBackendManifest{
		BaseURL:    fmt.Sprintf("http://127.0.0.1:%d", port),
		Token:      bm.token,
		Port:       port,
		PID:        pid,
		HermesRoot: bm.cfg.HermesRoot,
		HermesHome: bm.cfg.HermesHome,
		UpdatedAt:  time.Now().Format(time.RFC3339),
		Managed:    true,
	}
	return bm.writeManifest(manifest)
}

func loadManifestBackend(cfg Config) *backendInfo {
	path := filepath.Join(cfg.DataDir, desktopBackendManifestName)
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var manifest DesktopBackendManifest
	if err := json.Unmarshal(raw, &manifest); err != nil {
		return nil
	}
	port := manifest.Port
	if port <= 0 && manifest.BaseURL != "" {
		// Best-effort parse http://127.0.0.1:NNNN
		var parsed int
		if _, err := fmt.Sscanf(strings.TrimPrefix(manifest.BaseURL, "http://127.0.0.1:"), "%d", &parsed); err == nil {
			port = parsed
		}
	}
	if port <= 0 || isReservedOpsPort(port) {
		return nil
	}
	if manifest.PID > 0 && !processAlive(manifest.PID) {
		return nil
	}
	if !testBackendStatus(port) {
		return nil
	}
	return &backendInfo{PID: uint32(manifest.PID), Port: port, Cmd: "manifest serve"}
}

func desktopLaunchEnv(cfg Config, manifest *DesktopBackendManifest) []string {
	env := []string{
		"HERMES_HOME=" + cfg.HermesHome,
		"HERMES_DESKTOP_HERMES_ROOT=" + cfg.HermesRoot,
		"HERMES_DESKTOP_CWD=" + cfg.HermesRoot,
	}
	webDist := resolveWebDist(cfg.HermesRoot)
	if webDist != "" {
		env = append(env, "HERMES_DESKTOP_DASHBOARD_WEB_DIST="+webDist)
	}
	if manifest != nil && manifest.BaseURL != "" && manifest.Token != "" {
		env = append(env,
			"HERMES_DESKTOP_REMOTE_URL="+manifest.BaseURL,
			"HERMES_DESKTOP_REMOTE_TOKEN="+manifest.Token,
		)
	}
	return env
}
