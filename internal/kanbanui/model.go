package kanbanui

import (
	"time"

	"github.com/nous-research/hermes-agent/internal/kanbanui/logviewer"
)

// Config holds runtime options for the kanban TUI.
type Config struct {
	DBPath          string
	RefreshInterval time.Duration
}

// View represents the current screen in the TUI.
type View int

const (
	ViewBoard View = iota
	ViewLogs
)

// Model is the Bubble Tea application state for the kanban board UI.
type Model struct {
	config Config
	store  Store
	keymap KeyMap
	styles Styles

	width   int
	height  int
	cursor  int
	columns []Column
	tasks   []Task
	status  string

	// Log viewer integration
	currentView View
	logViewer   logviewer.Model
	logChan     chan logviewer.Entry
	taskLogMsg  chan string // incoming log lines from tailer
	tailerDone  chan struct{} // signals tailer goroutine to stop
	viewingTask string        // task ID currently being viewed
}

// New creates a kanban UI model wired to the SQLite store.
func New(config Config) *Model {
	if config.RefreshInterval <= 0 {
		config.RefreshInterval = 2 * time.Second
	}

	store := NewStore(config.DBPath)
	return &Model{
		config:      config,
		store:       store,
		keymap:      DefaultKeyMap(),
		styles:      DefaultStyles(),
		currentView: ViewBoard,
		logChan:     make(chan logviewer.Entry, 100),
		taskLogMsg:  make(chan string, 100),
		tailerDone:  make(chan struct{}),
	}
}
