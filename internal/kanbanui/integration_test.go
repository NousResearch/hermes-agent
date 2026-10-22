package kanbanui

import (
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/nous-research/hermes-agent/internal/kanbanui/logviewer"
)

// --- Test helpers for events ---

func setupEventsTable(t *testing.T, s Store) {
	t.Helper()
	if s.db == nil {
		t.Fatal("store has no database")
	}
	_, err := s.db.Exec(`
		CREATE TABLE task_events (
			task_id TEXT NOT NULL,
			kind TEXT NOT NULL,
			payload TEXT,
			created_at INTEGER NOT NULL,
			run_id INTEGER
		)
	`)
	if err != nil {
		t.Fatalf("create events table: %v", err)
	}
}

func insertEvent(t *testing.T, s Store, taskID, kind string, payload string, ts time.Time, runID int) {
	t.Helper()
	if s.db == nil {
		t.Fatal("store has no database")
	}
	_, err := s.db.Exec(
		"INSERT INTO task_events (task_id, kind, payload, created_at, run_id) VALUES (?, ?, ?, ?, ?)",
		taskID, kind, payload, ts.Unix(), runID,
	)
	if err != nil {
		t.Fatalf("insert event: %v", err)
	}
}

// --- TaskEvent Tests ---

func TestTaskEventMessage(t *testing.T) {
	ts := time.Date(2025, 1, 1, 14, 30, 45, 0, time.UTC)
	ev := TaskEvent{
		TaskID:    "t1",
		Kind:      "created",
		CreatedAt: ts,
		RunID:     42,
	}

	msg := ev.Message()
	if !strings.Contains(msg, "14:30:45") {
		t.Errorf("Message() = %q, want timestamp 14:30:45", msg)
	}
	if !strings.Contains(msg, "created") {
		t.Errorf("Message() = %q, want kind 'created'", msg)
	}
	if !strings.Contains(msg, "#42") {
		t.Errorf("Message() = %q, want run #42", msg)
	}
}

// --- LoadTaskEvents Tests ---

func TestLoadTaskEventsEmpty(t *testing.T) {
	store := newInMemoryStore(t)
	defer store.Close()
	setupTestSchema(t, store)
	setupEventsTable(t, store)

	events := store.LoadTaskEvents("t1", time.Time{})
	if len(events) != 0 {
		t.Errorf("LoadTaskEvents() = %d events, want 0", len(events))
	}
}

func TestLoadTaskEventsNilDB(t *testing.T) {
	store := Store{dbPath: "nope"}
	events := store.LoadTaskEvents("t1", time.Time{})
	if events != nil {
		t.Errorf("LoadTaskEvents on nil db = %v, want nil", events)
	}
}

func TestLoadTaskEventsAll(t *testing.T) {
	store := newInMemoryStore(t)
	defer store.Close()
	setupTestSchema(t, store)
	setupEventsTable(t, store)

	ts1 := time.Date(2025, 1, 1, 10, 0, 0, 0, time.UTC)
	ts2 := time.Date(2025, 1, 1, 11, 0, 0, 0, time.UTC)
	ts3 := time.Date(2025, 1, 1, 12, 0, 0, 0, time.UTC)

	insertEvent(t, store, "t1", "created", "", ts1, 1)
	insertEvent(t, store, "t1", "claimed", "", ts2, 2)
	insertEvent(t, store, "t1", "completed", "", ts3, 2)

	events := store.LoadTaskEvents("t1", time.Time{})
	if len(events) != 3 {
		t.Fatalf("LoadTaskEvents() = %d events, want 3", len(events))
	}

	if events[0].Kind != "created" {
		t.Errorf("events[0].Kind = %q, want 'created'", events[0].Kind)
	}
	if events[1].Kind != "claimed" {
		t.Errorf("events[1].Kind = %q, want 'claimed'", events[1].Kind)
	}
	if events[2].Kind != "completed" {
		t.Errorf("events[2].Kind = %q, want 'completed'", events[2].Kind)
	}
}

func TestLoadTaskEventsSince(t *testing.T) {
	store := newInMemoryStore(t)
	defer store.Close()
	setupTestSchema(t, store)
	setupEventsTable(t, store)

	ts1 := time.Date(2025, 1, 1, 10, 0, 0, 0, time.UTC)
	ts2 := time.Date(2025, 1, 1, 11, 0, 0, 0, time.UTC)
	ts3 := time.Date(2025, 1, 1, 12, 0, 0, 0, time.UTC)

	insertEvent(t, store, "t1", "created", "", ts1, 1)
	insertEvent(t, store, "t1", "claimed", "", ts2, 2)
	insertEvent(t, store, "t1", "completed", "", ts3, 2)

	since := ts2
	events := store.LoadTaskEvents("t1", since)
	if len(events) != 1 {
		t.Fatalf("LoadTaskEvents(since=%v) = %d events, want 1", since, len(events))
	}
	if events[0].Kind != "completed" {
		t.Errorf("events[0].Kind = %q, want 'completed'", events[0].Kind)
	}
}

func TestLoadTaskEventsOtherTask(t *testing.T) {
	store := newInMemoryStore(t)
	defer store.Close()
	setupTestSchema(t, store)
	setupEventsTable(t, store)

	ts := time.Date(2025, 1, 1, 10, 0, 0, 0, time.UTC)
	insertEvent(t, store, "t1", "created", "", ts, 1)
	insertEvent(t, store, "t2", "created", "", ts, 1)

	events := store.LoadTaskEvents("t1", time.Time{})
	if len(events) != 1 {
		t.Fatalf("LoadTaskEvents(t1) = %d events, want 1", len(events))
	}
	if events[0].TaskID != "t1" {
		t.Errorf("events[0].TaskID = %q, want 't1'", events[0].TaskID)
	}
}

func TestLoadTaskEventsWithPayload(t *testing.T) {
	store := newInMemoryStore(t)
	defer store.Close()
	setupTestSchema(t, store)
	setupEventsTable(t, store)

	ts := time.Date(2025, 1, 1, 10, 0, 0, 0, time.UTC)
	insertEvent(t, store, "t1", "claimed", `{"by":"coder"}`, ts, 5)

	events := store.LoadTaskEvents("t1", time.Time{})
	if len(events) != 1 {
		t.Fatalf("LoadTaskEvents() = %d events, want 1", len(events))
	}
	if events[0].Payload != `{"by":"coder"}` {
		t.Errorf("events[0].Payload = %q, want JSON payload", events[0].Payload)
	}
	if events[0].RunID != 5 {
		t.Errorf("events[0].RunID = %d, want 5", events[0].RunID)
	}
}

// --- Model Integration Tests ---

func TestModelInit(t *testing.T) {
	config := Config{
		DBPath:          ":memory:",
		RefreshInterval: time.Second,
	}
	m := New(config)

	if m.currentView != ViewBoard {
		t.Errorf("currentView = %v, want ViewBoard", m.currentView)
	}
	if m.viewingTask != "" {
		t.Errorf("viewingTask = %q, want empty", m.viewingTask)
	}
	if m.logChan == nil {
		t.Error("logChan is nil")
	}
	if m.taskLogMsg == nil {
		t.Error("taskLogMsg is nil")
	}
}

func TestModelInitDefaultRefresh(t *testing.T) {
	config := Config{DBPath: ":memory:"}
	m := New(config)

	if m.config.RefreshInterval != 2*time.Second {
		t.Errorf("RefreshInterval = %v, want 2s", m.config.RefreshInterval)
	}
}

func TestModelViewBoardDefault(t *testing.T) {
	config := Config{DBPath: ":memory:"}
	m := New(config)

	view := m.View()
	if !strings.Contains(view, "Hermes Kanban TUI") {
		t.Errorf("View() = %q, want board header", view)
	}
}

func TestModelOpenLogViewerNoTasks(t *testing.T) {
	config := Config{DBPath: ":memory:"}
	m := New(config)

	result, cmd := m.openLogViewer()
	if result.currentView != ViewBoard {
		t.Errorf("openLogViewer() with no tasks: view = %v, want ViewBoard", result.currentView)
	}
	if cmd != nil {
		t.Error("openLogViewer() with no tasks should return nil cmd")
	}
	if result.status != "no task selected" {
		t.Errorf("status = %q, want 'no task selected'", result.status)
	}
}

func TestModelCloseLogViewer(t *testing.T) {
	config := Config{DBPath: ":memory:"}
	m := New(config)
	m.currentView = ViewLogs
	m.viewingTask = "t1"

	result, _ := m.closeLogViewer()
	if result.currentView != ViewBoard {
		t.Errorf("closeLogViewer(): view = %v, want ViewBoard", result.currentView)
	}
	if result.viewingTask != "" {
		t.Errorf("closeLogViewer(): viewingTask = %q, want empty", result.viewingTask)
	}
}

func TestModelHandleLogKeysEsc(t *testing.T) {
	config := Config{DBPath: ":memory:"}
	m := New(config)
	m.currentView = ViewLogs
	m.viewingTask = "t1"

	escMsg := tea.KeyMsg{Type: tea.KeyEsc}
	result, _ := m.handleLogKeys(escMsg)
	m2 := result.(*Model)

	if m2.currentView != ViewBoard {
		t.Errorf("handleLogKeys(Esc): view = %v, want ViewBoard", m2.currentView)
	}
}

func TestModelHandleLogKeysBack(t *testing.T) {
	config := Config{DBPath: ":memory:"}
	m := New(config)
	m.currentView = ViewLogs
	m.viewingTask = "t1"

	bMsg := tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'b'}}
	result, _ := m.handleLogKeys(bMsg)
	m2 := result.(*Model)

	if m2.currentView != ViewBoard {
		t.Errorf("handleLogKeys('b'): view = %v, want ViewBoard", m2.currentView)
	}
}

func TestModelHandleBoardKeysQuit(t *testing.T) {
	config := Config{DBPath: ":memory:"}
	m := New(config)

	quitMsg := tea.KeyMsg{Type: tea.KeyCtrlC}
	_, cmd := m.handleBoardKeys(quitMsg)

	if cmd == nil {
		t.Error("handleBoardKeys(C-c) should return tea.Quit cmd")
	}
}

func TestModelHandleBoardKeysUp(t *testing.T) {
	config := Config{DBPath: ":memory:"}
	m := New(config)
	m.tasks = []Task{
		{ID: "t1", Title: "First"},
		{ID: "t2", Title: "Second"},
		{ID: "t3", Title: "Third"},
	}
	m.cursor = 2

	upMsg := tea.KeyMsg{Type: tea.KeyUp}
	result, _ := m.handleBoardKeys(upMsg)
	m2 := result.(*Model)

	if m2.cursor != 1 {
		t.Errorf("handleBoardKeys(Up): cursor = %d, want 1", m2.cursor)
	}
}

func TestModelHandleBoardKeysDown(t *testing.T) {
	config := Config{DBPath: ":memory:"}
	m := New(config)
	m.tasks = []Task{
		{ID: "t1", Title: "First"},
		{ID: "t2", Title: "Second"},
	}
	m.cursor = 0

	downMsg := tea.KeyMsg{Type: tea.KeyDown}
	result, _ := m.handleBoardKeys(downMsg)
	m2 := result.(*Model)

	if m2.cursor != 1 {
		t.Errorf("handleBoardKeys(Down): cursor = %d, want 1", m2.cursor)
	}
}

func TestModelHandleBoardKeysDownAtBottom(t *testing.T) {
	config := Config{DBPath: ":memory:"}
	m := New(config)
	m.tasks = []Task{{ID: "t1", Title: "Only"}}
	m.cursor = 0

	downMsg := tea.KeyMsg{Type: tea.KeyDown}
	result, _ := m.handleBoardKeys(downMsg)
	m2 := result.(*Model)

	if m2.cursor != 0 {
		t.Errorf("handleBoardKeys(Down) at bottom: cursor = %d, want 0", m2.cursor)
	}
}

func TestModelHandleBoardKeysUpAtTop(t *testing.T) {
	config := Config{DBPath: ":memory:"}
	m := New(config)
	m.tasks = []Task{{ID: "t1", Title: "Only"}}
	m.cursor = 0

	upMsg := tea.KeyMsg{Type: tea.KeyUp}
	result, _ := m.handleBoardKeys(upMsg)
	m2 := result.(*Model)

	if m2.cursor != 0 {
		t.Errorf("handleBoardKeys(Up) at top: cursor = %d, want 0", m2.cursor)
	}
}

func TestModelRenderLogView(t *testing.T) {
	config := Config{DBPath: ":memory:"}
	m := New(config)
	m.currentView = ViewLogs
	m.viewingTask = "t123"
	m.tasks = []Task{{ID: "t123", Title: "Test Task"}}
	m.logViewer = logviewer.New(80, 20)

	view := m.renderLogView()
	if !strings.Contains(view, "t123") {
		t.Errorf("renderLogView() = %q, want task ID in header", view)
	}
	if !strings.Contains(view, "Test Task") {
		t.Errorf("renderLogView() = %q, want task title in header", view)
	}
	if !strings.Contains(view, "esc/b: back to board") {
		t.Errorf("renderLogView() = %q, want footer hint", view)
	}
}

func TestModelRenderLogViewNoTitle(t *testing.T) {
	config := Config{DBPath: ":memory:"}
	m := New(config)
	m.currentView = ViewLogs
	m.viewingTask = "t_unknown"
	m.tasks = []Task{}
	m.logViewer = logviewer.New(80, 20)

	view := m.renderLogView()
	if !strings.Contains(view, "t_unknown") {
		t.Errorf("renderLogView() = %q, want task ID", view)
	}
}

func TestModelUpdateTaskLogMsg(t *testing.T) {
	config := Config{DBPath: ":memory:"}
	m := New(config)
	m.currentView = ViewLogs
	m.viewingTask = "t1"
	m.logViewer = logviewer.New(80, 20)

	logMsg := TaskLogMsg{
		TaskID: "t1",
		Entry:  logviewer.NewEntry(logviewer.LevelInfo, "test log line"),
	}

	result, _ := m.Update(logMsg)
	m2 := result.(*Model)

	view := m2.logViewer.View()
	if !strings.Contains(view, "test log line") {
		t.Errorf("log viewer should contain 'test log line', got: %s", view)
	}
}

func TestModelUpdateTaskLogMsgWrongTask(t *testing.T) {
	config := Config{DBPath: ":memory:"}
	m := New(config)
	m.currentView = ViewLogs
	m.viewingTask = "t1"
	m.logViewer = logviewer.New(80, 20)

	logMsg := TaskLogMsg{
		TaskID: "t2",
		Entry:  logviewer.NewEntry(logviewer.LevelInfo, "other task log"),
	}

	result, _ := m.Update(logMsg)
	m2 := result.(*Model)

	view := m2.logViewer.View()
	if strings.Contains(view, "other task log") {
		t.Error("log viewer should NOT contain 'other task log' (wrong task)")
	}
}

func TestModelUpdateTaskLogMsgBoardView(t *testing.T) {
	config := Config{DBPath: ":memory:"}
	m := New(config)
	m.currentView = ViewBoard
	m.logViewer = logviewer.New(80, 20)

	logMsg := TaskLogMsg{
		TaskID: "t1",
		Entry:  logviewer.NewEntry(logviewer.LevelInfo, "should be ignored"),
	}

	result, _ := m.Update(logMsg)
	m2 := result.(*Model)

	view := m2.logViewer.View()
	if strings.Contains(view, "should be ignored") {
		t.Error("log viewer should NOT contain entry when in board view")
	}
}

// --- KeyMap Tests ---

func TestKeyMapHasLog(t *testing.T) {
	km := DefaultKeyMap()

	logKeys := km.Log.Keys()
	hasL := false
	for _, k := range logKeys {
		if k == "l" {
			hasL = true
			break
		}
	}
	if !hasL {
		t.Errorf("Log key binding keys = %v, want 'l' included", logKeys)
	}
}

func TestKeyMapShortHelp(t *testing.T) {
	km := DefaultKeyMap()
	help := km.ShortHelp()

	foundLog := false
	for _, b := range help {
		for _, k := range b.Keys() {
			if k == "l" {
				foundLog = true
				break
			}
		}
	}
	if !foundLog {
		t.Error("ShortHelp() should include Log binding")
	}
}

// --- View Constants Tests ---

func TestViewConstants(t *testing.T) {
	if ViewBoard != 0 {
		t.Errorf("ViewBoard = %d, want 0", ViewBoard)
	}
	if ViewLogs != 1 {
		t.Errorf("ViewLogs = %d, want 1", ViewLogs)
	}
}
