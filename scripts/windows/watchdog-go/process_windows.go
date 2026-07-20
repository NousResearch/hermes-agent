//go:build windows

package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/yusufpapurcu/wmi"
)

type win32Process struct {
	ProcessID   uint32
	Name        string
	CommandLine string
}

func getDesktopProcesses() ([]win32Process, error) {
	var procs []win32Process
	err := wmi.Query("SELECT ProcessId, Name, CommandLine FROM Win32_Process WHERE Name = 'Hermes.exe'", &procs)
	return procs, err
}

// reservedOpsPorts are stack-owned listeners — never treat as Desktop's ephemeral hermes serve.
var reservedOpsPorts = map[int]struct{}{
	8080: {}, 8081: {}, 8646: {}, 8765: {}, 8787: {}, 9119: {}, 9120: {}, 9920: {}, 18794: {},
}

func isReservedOpsPort(port int) bool {
	_, ok := reservedOpsPorts[port]
	return ok
}

func isDesktopBackendCommandLine(cl string) bool {
	if cl == "" {
		return false
	}
	lower := strings.ToLower(cl)
	if !strings.Contains(cl, "hermes_cli.main") &&
		!strings.Contains(cl, "\\hermes.exe") &&
		!strings.Contains(cl, "Scripts\\hermes.exe") {
		return false
	}
	// Never manage gateway / harness / cron — those are stack services.
	if strings.Contains(lower, " gateway") || strings.Contains(lower, " harness") || strings.Contains(lower, " cron") {
		return false
	}
	// Explicit ops dashboard / fixed ports are not Desktop-spawned backends.
	if strings.Contains(cl, "--port 9120") || strings.Contains(cl, "--port=9120") ||
		strings.Contains(cl, "--port 8787") || strings.Contains(cl, "--port=8787") {
		return false
	}
	if strings.Contains(cl, " serve") || strings.Contains(cl, "\tserve") {
		// Prefer Desktop's ephemeral serve (--port 0). Bare "serve" still matches,
		// but find/reap skip reserved ops ports so dashboard:9120 is never claimed/killed.
		return true
	}
	if strings.Contains(cl, "dashboard") && strings.Contains(cl, "--no-open") {
		return true
	}
	return false
}

func getDesktopBackendCandidates() ([]win32Process, error) {
	var all []win32Process
	if err := wmi.Query("SELECT ProcessId, Name, CommandLine FROM Win32_Process", &all); err != nil {
		return nil, err
	}
	out := make([]win32Process, 0, 4)
	for _, p := range all {
		if isDesktopBackendCommandLine(p.CommandLine) {
			out = append(out, p)
		}
	}
	return out, nil
}

func getListeningPorts(pid uint32) ([]int, error) {
	out, err := exec.Command("netstat", "-ano", "-p", "tcp").Output()
	if err != nil {
		return nil, err
	}
	ports := make([]int, 0, 2)
	target := fmt.Sprintf("%d", pid)
	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)
		if !strings.Contains(line, "LISTENING") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 5 || fields[len(fields)-1] != target {
			continue
		}
		hostPort := fields[1]
		idx := strings.LastIndex(hostPort, ":")
		if idx < 0 {
			continue
		}
		portStr := hostPort[idx+1:]
		port, convErr := strconv.Atoi(portStr)
		if convErr == nil && port > 0 {
			ports = appendUniqueInt(ports, port)
		}
	}
	return ports, nil
}

func appendUniqueInt(list []int, v int) []int {
	for _, existing := range list {
		if existing == v {
			return list
		}
	}
	return append(list, v)
}

func testBackendStatus(port int) bool {
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(fmt.Sprintf("http://127.0.0.1:%d/api/status", port))
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

type backendInfo struct {
	PID  uint32 `json:"pid"`
	Port int    `json:"port"`
	Cmd  string `json:"cmd,omitempty"`
}

func findHealthyDesktopBackend() *backendInfo {
	candidates, err := getDesktopBackendCandidates()
	if err != nil {
		return nil
	}
	for _, proc := range candidates {
		ports, perr := getListeningPorts(proc.ProcessID)
		if perr != nil {
			continue
		}
		for _, port := range ports {
			if isReservedOpsPort(port) {
				continue
			}
			if testBackendStatus(port) {
				return &backendInfo{
					PID:  proc.ProcessID,
					Port: port,
					Cmd:  proc.CommandLine,
				}
			}
		}
	}
	return nil
}

func stopProcessPID(pid uint32) {
	// /T reaps the process tree. Plain /F leaves Electron grandchildren and
	// desktop-spawned hermes serve orphans (before-quit never runs on force kill).
	_ = exec.Command("taskkill", "/PID", fmt.Sprintf("%d", pid), "/T", "/F").Run()
}

func stopAllDesktopProcessTrees(logger *Logger) {
	desktop, err := getDesktopProcesses()
	if err != nil {
		logger.Infof("enumerate Hermes.exe for tree-kill: %v", err)
	}
	seen := make(map[uint32]struct{}, len(desktop))
	for _, p := range desktop {
		if _, ok := seen[p.ProcessID]; ok {
			continue
		}
		seen[p.ProcessID] = struct{}{}
		logger.Infof("tree-killing Hermes.exe pid=%d", p.ProcessID)
		stopProcessPID(p.ProcessID)
	}
	// Catch any helper that WMI missed under the packaged image name.
	_ = exec.Command("taskkill", "/IM", "Hermes.exe", "/T", "/F").Run()
}

func stopOrphanDesktopBackends(logger *Logger, cfg Config, skipPIDs ...uint32) int {
	desktop, err := getDesktopProcesses()
	if err == nil && len(desktop) > 0 {
		return 0
	}
	skip := make(map[uint32]struct{}, len(skipPIDs))
	for _, pid := range skipPIDs {
		if pid > 0 {
			skip[pid] = struct{}{}
		}
	}
	skipPort := cfg.ManagedBackendPort
	if skipPort <= 0 {
		skipPort = DefaultManagedBackendPort
	}
	candidates, err := getDesktopBackendCandidates()
	if err != nil {
		return 0
	}
	n := 0
	for _, proc := range candidates {
		if _, keep := skip[proc.ProcessID]; keep {
			logger.Infof("skip reap pid=%d (managed backend)", proc.ProcessID)
			continue
		}
		ports, _ := getListeningPorts(proc.ProcessID)
		skipProc := false
		for _, port := range ports {
			if port == skipPort {
				logger.Infof("skip reap pid=%d (managed port %d)", proc.ProcessID, port)
				skipProc = true
				break
			}
			if isReservedOpsPort(port) {
				logger.Infof("skip reap pid=%d (ops port %d)", proc.ProcessID, port)
				skipProc = true
				break
			}
		}
		if skipProc {
			continue
		}
		logger.Infof("reaping orphan backend pid=%d", proc.ProcessID)
		stopProcessPID(proc.ProcessID)
		n++
	}
	return n
}

func readLaunchManifest(cfg Config, bm *BackendManager) *DesktopBackendManifest {
	if bm != nil {
		if manifest, err := bm.readManifest(); err == nil && manifest != nil {
			return manifest
		}
	}
	path := filepath.Join(cfg.DataDir, desktopBackendManifestName)
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var manifest DesktopBackendManifest
	if json.Unmarshal(raw, &manifest) != nil {
		return nil
	}
	if manifest.BaseURL == "" || manifest.Token == "" {
		return nil
	}
	return &manifest
}

func startPackagedDesktop(cfg Config, logger *Logger, bm *BackendManager) bool {
	if !fileExists(cfg.PackagedExe) {
		logger.Infof("Hermes.exe missing at %s", cfg.PackagedExe)
		return false
	}
	work := filepath.Dir(cfg.PackagedExe)
	cmd := exec.Command(cfg.PackagedExe)
	cmd.Dir = work
	manifest := readLaunchManifest(cfg, bm)
	cmd.Env = append(os.Environ(), desktopLaunchEnv(cfg, manifest)...)
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true}
	if err := cmd.Start(); err != nil {
		logger.Infof("failed to launch Desktop: %v", err)
		return false
	}
	if manifest != nil {
		logger.Infof("launched %s (prewarmed backend %s)", cfg.PackagedExe, manifest.BaseURL)
	} else {
		logger.Infof("launched %s", cfg.PackagedExe)
	}
	return true
}

func restartPackagedDesktop(cfg Config, logger *Logger, bm *BackendManager) bool {
	logger.Infof("restarting Desktop (force backend respawn)")
	stopAllDesktopProcessTrees(logger)
	time.Sleep(2 * time.Second)
	var skipPID uint32
	if bm != nil {
		if managed := bm.currentHealthy(); managed != nil {
			skipPID = managed.PID
		}
	}
	// Desktop is gone — reap leftover ephemeral serves (managed :9118 is skipped).
	stopOrphanDesktopBackends(logger, cfg, skipPID)
	time.Sleep(1 * time.Second)
	if bm != nil {
		if _, err := bm.EnsureHealthy(); err != nil {
			logger.Infof("pre-restart managed backend: %v", err)
		}
	}
	return startPackagedDesktop(cfg, logger, bm)
}
