// Hermes Desktop↔backend mutual watchdog (Windows).
//
// ISOLATION: standalone operator binary — NOT registered in Hermes plugins,
// tools, skills, MCP, or cron. Mutating HTTP APIs require HERMES_WATCHDOG_ADMIN_TOKEN.
package main

import (
	"flag"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"tailscale.com/tsnet"
)

func main() {
	if runtime.GOOS != "windows" {
		log.Fatalf("hermes-watchdog supports Windows only (got %s)", runtime.GOOS)
	}

	repoRoot := flag.String("hermes-root", "", "Hermes repo root (default: auto from exe path)")
	hermesHome := flag.String("hermes-home", defaultHermesHome(), "HERMES_HOME profile directory")
	packagedExe := flag.String("packaged-exe", "", "Packaged Hermes.exe path")
	dataDir := flag.String("data-dir", defaultDataDir(), "State directory (default %LOCALAPPDATA%\\HermesWatchdog)")
	listen := flag.String("listen", "127.0.0.1:9920", "Local HTTP listen address (empty disables)")
	tsnetHost := flag.String("tsnet-hostname", "hermes-watchdog", "Tailscale tsnet hostname")
	enableTsnet := flag.Bool("tsnet", false, "Enable Tailscale tsnet listener (also auto when authkey env set)")
	interval := flag.Int("interval", 20, "Watchdog probe interval seconds")
	failThreshold := flag.Int("fail-threshold", 2, "Consecutive backend failures before Desktop restart")
	prewarm := flag.Bool("prewarm-backend", true, "Pre-start and supervise a hermes serve for fast Desktop connect")
	backendStartTimeout := flag.Int("backend-start-timeout", 120, "Seconds to wait for managed serve /api/status")
	backendReadyTimeout := flag.Int("backend-ready-timeout", 45, "Extra seconds waiting for managed serve readiness")
	managedPort := flag.Int("managed-backend-port", DefaultManagedBackendPort, "Fixed localhost port for watchdog-managed hermes serve")
	once := flag.Bool("once", false, "Run a single watchdog cycle then exit")
	noHTTP := flag.Bool("no-http", false, "Disable HTTP control plane (watch loop only)")
	flag.Parse()

	root := sanitizePathFlag(*repoRoot)
	home := sanitizePathFlag(*hermesHome)
	if home == "" {
		home = defaultHermesHome()
	}
	if root == "" {
		root = detectRepoRoot()
	}
	if root != "" && !fileExists(filepath.Join(root, "pyproject.toml")) {
		if detected := detectRepoRoot(); detected != "" && fileExists(filepath.Join(detected, "pyproject.toml")) {
			root = detected
		}
	}
	// Paths with spaces (e.g. "...\New project\...") must stay intact; if a
	// broken partial Dir was ever used, prefer a root that still has pyproject.
	if root != "" && !dirExists(root) {
		if detected := detectRepoRoot(); detected != "" && dirExists(detected) {
			root = detected
		}
	}

	cfg := Config{
		IntervalSec:            *interval,
		FailThreshold:          *failThreshold,
		Once:                   *once,
		PrewarmBackend:         *prewarm,
		BackendStartTimeoutSec: *backendStartTimeout,
		BackendReadyTimeoutSec: *backendReadyTimeout,
		ManagedBackendPort:     *managedPort,
		ListenAddr:             strings.TrimSpace(*listen),
		TsnetHostname: *tsnetHost,
		EnableTsnet:   *enableTsnet,
		HermesRoot:    root,
		HermesHome:    home,
		PackagedExe:   *packagedExe,
		DataDir:       *dataDir,
		AdminToken:    loadAdminToken(),
		TsAuthKey:     loadTsAuthKey(),
	}
	if cfg.PackagedExe == "" {
		cfg.PackagedExe = defaultPackagedExe(root)
	}
	if cfg.TsAuthKey != "" {
		cfg.EnableTsnet = true
	}

	if err := ensureDir(cfg.DataDir); err != nil {
		log.Fatalf("data dir: %v", err)
	}
	logDir := filepath.Join(cfg.HermesHome, "logs")
	_ = ensureDir(logDir)
	cfg.LogPath = filepath.Join(logDir, "hermes-go-watchdog.log")
	cfg.LockPath = filepath.Join(cfg.DataDir, "watchdog.lock")
	cfg.StatePath = filepath.Join(cfg.DataDir, "watchdog.state.json")

	logger := NewLogger(cfg.LogPath)
	release, ok := acquireLock(cfg.LockPath, root, logger)
	if !ok {
		return
	}
	defer release()

	wd := NewWatchdog(cfg, logger)
	wd.PrewarmBackend()

	if *once {
		wd.RunCycle()
		logger.Infof("watchdog once complete")
		return
	}

	stop := make(chan struct{})
	shutdown := func() {
		select {
		case <-stop:
		default:
			close(stop)
		}
	}

	if !*noHTTP {
		srv := NewHTTPServer(cfg, wd, shutdown)
		handler := srv.Handler()
		if cfg.ListenAddr != "" {
			go serveHTTP(logger, "local", cfg.ListenAddr, handler)
		}
		if cfg.EnableTsnet {
			go serveTsnet(logger, cfg, handler)
		}
	}

	go wd.RunLoop(stop)
	<-stop
	logger.Infof("watchdog stop")
}

func serveHTTP(logger *Logger, label, addr string, handler http.Handler) {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		logger.Infof("%s listen failed on %s: %v", label, addr, err)
		return
	}
	logger.Infof("%s HTTP listening on %s", label, addr)
	s := &http.Server{Handler: handler}
	if err := s.Serve(ln); err != nil && err != http.ErrServerClosed {
		logger.Infof("%s HTTP server error: %v", label, err)
	}
}

func serveTsnet(logger *Logger, cfg Config, handler http.Handler) {
	srv := &tsnet.Server{
		Hostname: cfg.TsnetHostname,
		AuthKey:  cfg.TsAuthKey,
	}
	defer srv.Close()
	ln, err := srv.Listen("tcp", ":443")
	if err != nil {
		logger.Infof("tsnet listen failed: %v", err)
		return
	}
	logger.Infof("tsnet HTTP listening as %s (tailnet)", cfg.TsnetHostname)
	httpServer := &http.Server{Handler: handler}
	if err := httpServer.Serve(ln); err != nil && err != http.ErrServerClosed {
		logger.Infof("tsnet HTTP error: %v", err)
	}
}

func detectRepoRoot() string {
	exe, err := os.Executable()
	if err != nil {
		return ""
	}
	dir := filepath.Dir(exe)
	// dist → watchdog-go → windows → scripts → <repo>
	candidates := []string{
		filepath.Clean(filepath.Join(dir, "..", "..", "..", "..")),
		filepath.Clean(filepath.Join(dir, "..", "..", "..")),
	}
	for _, candidate := range candidates {
		if fileExists(filepath.Join(candidate, "pyproject.toml")) {
			return candidate
		}
	}
	if len(candidates) > 0 {
		return candidates[0]
	}
	return ""
}

// sanitizePathFlag strips accidental shell quotes and trims space so CreateProcess
// Dir never becomes `"C:\Users\...\New` (split at the space in "New project").
func sanitizePathFlag(raw string) string {
	s := strings.TrimSpace(raw)
	s = strings.Trim(s, `"'`)
	return strings.TrimSpace(s)
}

func dirExists(path string) bool {
	if path == "" {
		return false
	}
	st, err := os.Stat(path)
	return err == nil && st.IsDir()
}
