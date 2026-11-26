package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/nous-research/hermes-agent/internal/kanbanui"
)

const version = "0.0.1"

func main() {
	fs := flag.NewFlagSet("hermes-kanban-tui", flag.ExitOnError)
	fs.SetOutput(os.Stderr)

	defaultDB := defaultKanbanDBPath()
	dbPath := fs.String("db", defaultDB, "path to Hermes kanban SQLite database")
	refresh := fs.Duration("refresh", 2*time.Second, "refresh interval for polling the kanban database")
	noAltScreen := fs.Bool("no-alt-screen", false, "disable alternate screen rendering")
	showVersion := fs.Bool("version", false, "print version and exit")

	fs.Usage = func() {
		printUsage(fs, os.Stderr, defaultDB)
	}

	if err := fs.Parse(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(2)
	}

	if *showVersion {
		fmt.Printf("hermes-kanban-tui %s\n", version)
		return
	}

	if *refresh <= 0 {
		fmt.Fprintln(os.Stderr, "error: --refresh must be greater than zero")
		os.Exit(2)
	}

	model := kanbanui.New(kanbanui.Config{
		DBPath:          *dbPath,
		RefreshInterval: *refresh,
	})

	opts := []tea.ProgramOption{}
	if !*noAltScreen {
		opts = append(opts, tea.WithAltScreen())
	}

	program := tea.NewProgram(model, opts...)
	if _, err := program.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func defaultKanbanDBPath() string {
	if value := os.Getenv("HERMES_KANBAN_DB"); value != "" {
		return value
	}
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		return filepath.Join(home, ".hermes", "kanban.db")
	}
	return "kanban.db"
}

func printUsage(fs *flag.FlagSet, w io.Writer, defaultDB string) {
	fmt.Fprintf(w, "Hermes Kanban TUI\n\n")
	fmt.Fprintf(w, "Usage:\n  %s [flags]\n\n", fs.Name())
	fmt.Fprintf(w, "Flags:\n")
	fmt.Fprintf(w, "  --db PATH\n        Path to the Hermes kanban SQLite database.\n        Default: %s\n", defaultDB)
	fmt.Fprintf(w, "  --refresh DURATION\n        Refresh interval for reloading board state, such as 500ms, 2s, or 1m.\n        Default: 2s\n")
	fmt.Fprintf(w, "  --no-alt-screen\n        Disable alternate screen rendering for logs, tests, and headless runs.\n")
	fmt.Fprintf(w, "  --version\n        Print version and exit.\n")
	fmt.Fprintf(w, "  --help\n        Show this help message.\n")
}
