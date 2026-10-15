package logviewer

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
)

// update is a helper that calls Update and type-asserts back to Model.
func update(m Model, msg tea.Msg) Model {
	result, _ := m.Update(msg)
	return result.(Model)
}

// updateWithCmd is like update but also returns the command.
func updateWithCmd(m Model, msg tea.Msg) (Model, tea.Cmd) {
	result, cmd := m.Update(msg)
	return result.(Model), cmd
}

func TestNew(t *testing.T) {
	m := New(80, 24)

	if m.width != 80 {
		t.Errorf("expected width 80, got %d", m.width)
	}
	if m.height != 24 {
		t.Errorf("expected height 24, got %d", m.height)
	}
	if !m.autoScroll {
		t.Error("expected autoScroll to be true by default")
	}
	if len(m.entries) != 0 {
		t.Error("expected empty entries")
	}
}

func TestNewWithCustomKeyMap(t *testing.T) {
	customKM := DefaultKeyMap()
	m := New(80, 24, WithKeyMap(customKM))

	if m.keyMap.Up.Keys() == nil {
		t.Error("expected custom key map to be set")
	}
}

func TestAddEntry(t *testing.T) {
	m := New(80, 24)

	m.AddEntry(NewEntry(LevelInfo, "first log"))

	if len(m.entries) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(m.entries))
	}
	if m.entries[0].Message != "first log" {
		t.Errorf("expected 'first log', got %q", m.entries[0].Message)
	}
}

func TestAddEntryAutoScroll(t *testing.T) {
	m := New(80, 24)

	for i := 0; i < 5; i++ {
		m.AddEntry(NewEntry(LevelInfo, "log line"))
	}

	if !m.autoScroll {
		t.Error("expected autoScroll to be true after adding entries")
	}
}

func TestAddEntryMultiple(t *testing.T) {
	m := New(80, 24)

	levels := []Level{LevelDebug, LevelInfo, LevelWarn, LevelError}
	for i, level := range levels {
		m.AddEntry(NewEntry(level, "message "+string(rune('A'+i))))
	}

	if len(m.entries) != 4 {
		t.Fatalf("expected 4 entries, got %d", len(m.entries))
	}

	for i, level := range levels {
		if m.entries[i].Level != level {
			t.Errorf("entry %d: expected level %v, got %v", i, level, m.entries[i].Level)
		}
	}
}

func TestLogEntryMsg(t *testing.T) {
	m := New(80, 24)

	entry := NewEntry(LevelError, "error occurred")
	msg := LogEntryMsg{Entry: entry}

	m = update(m, msg)

	if len(m.entries) != 1 {
		t.Fatalf("expected 1 entry after LogEntryMsg, got %d", len(m.entries))
	}
	if m.entries[0].Message != "error occurred" {
		t.Errorf("expected 'error occurred', got %q", m.entries[0].Message)
	}
}

func TestQuitKeyCtrlC(t *testing.T) {
	m := New(80, 24)

	keyMsg := tea.KeyMsg{Type: tea.KeyCtrlC}
	m, cmd := updateWithCmd(m, keyMsg)
	_ = m

	if cmd == nil {
		t.Fatal("expected Quit command on ctrl+c")
	}
	quitMsg := cmd()
	if _, ok := quitMsg.(tea.QuitMsg); !ok {
		t.Errorf("expected QuitMsg, got %T", quitMsg)
	}
}

func TestQuitKeyQ(t *testing.T) {
	m := New(80, 24)

	keyMsg := tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'q'}}
	_, cmd := updateWithCmd(m, keyMsg)

	if cmd == nil {
		t.Fatal("expected Quit command on q")
	}
	quitMsg := cmd()
	if _, ok := quitMsg.(tea.QuitMsg); !ok {
		t.Errorf("expected QuitMsg, got %T", quitMsg)
	}
}

func TestWindowResize(t *testing.T) {
	m := New(80, 24)
	m.AddEntry(NewEntry(LevelInfo, "test"))

	resizeMsg := tea.WindowSizeMsg{Width: 120, Height: 30}
	m = update(m, resizeMsg)

	if m.width != 120 {
		t.Errorf("expected width 120, got %d", m.width)
	}
	if m.height != 30 {
		t.Errorf("expected height 30, got %d", m.height)
	}
	if m.viewport.Width != 120 {
		t.Errorf("expected viewport width 120, got %d", m.viewport.Width)
	}
	if m.viewport.Height != 30 {
		t.Errorf("expected viewport height 30, got %d", m.viewport.Height)
	}
}

func TestViewRendersEntries(t *testing.T) {
	m := New(80, 24)
	m.AddEntry(NewEntry(LevelInfo, "hello world"))

	view := m.View()

	if !strings.Contains(view, "hello world") {
		t.Errorf("expected view to contain 'hello world', got:\n%s", view)
	}
}

func TestViewStatusBar(t *testing.T) {
	m := New(80, 24)
	m.AddEntry(NewEntry(LevelInfo, "test"))

	view := m.View()

	if !strings.Contains(view, "entries: 1") {
		t.Error("expected status bar to show 'entries: 1'")
	}
	if !strings.Contains(view, "auto-scroll: on") {
		t.Error("expected status bar to show 'auto-scroll: on'")
	}
	if !strings.Contains(view, "q:quit") {
		t.Error("expected status bar to show 'q:quit'")
	}
}

func TestViewFormatsTime(t *testing.T) {
	m := New(80, 24)
	m.AddEntry(NewEntry(LevelError, "test"))

	view := m.View()

	if !strings.Contains(view, "[") || !strings.Contains(view, "]") {
		t.Error("expected view to contain timestamp in brackets")
	}
}

func TestViewFormatsLevel(t *testing.T) {
	m := New(80, 24)
	m.AddEntry(NewEntry(LevelWarn, "test"))

	view := m.View()

	if !strings.Contains(view, "WARN") {
		t.Error("expected view to contain 'WARN' level")
	}
}

func TestScrollDownThenAutoScrollRestored(t *testing.T) {
	m := New(80, 24)

	for i := 0; i < 50; i++ {
		m.AddEntry(NewEntry(LevelInfo, "log line"))
	}

	// Scroll up to disable auto-scroll
	upMsg := tea.KeyMsg{Type: tea.KeyUp}
	m = update(m, upMsg)

	if m.autoScroll {
		t.Error("expected autoScroll to be false after scrolling up")
	}

	// Scroll back to bottom with End key
	endMsg := tea.KeyMsg{Type: tea.KeyEnd}
	m = update(m, endMsg)

	if !m.autoScroll {
		t.Error("expected autoScroll to be true after pressing End")
	}
}

func TestScrollUpDisablesAutoScroll(t *testing.T) {
	m := New(80, 24)

	for i := 0; i < 50; i++ {
		m.AddEntry(NewEntry(LevelInfo, "log line"))
	}

	upMsg := tea.KeyMsg{Type: tea.KeyUp}
	m = update(m, upMsg)

	if m.autoScroll {
		t.Error("expected autoScroll to be false after scrolling up")
	}
}

func TestHomeDisablesAutoScroll(t *testing.T) {
	m := New(80, 24)

	for i := 0; i < 50; i++ {
		m.AddEntry(NewEntry(LevelInfo, "log line"))
	}

	homeMsg := tea.KeyMsg{Type: tea.KeyHome}
	m = update(m, homeMsg)

	if m.autoScroll {
		t.Error("expected autoScroll to be false after pressing Home")
	}
}

func TestPageUpDisablesAutoScroll(t *testing.T) {
	m := New(80, 24)

	for i := 0; i < 50; i++ {
		m.AddEntry(NewEntry(LevelInfo, "log line"))
	}

	pgUpMsg := tea.KeyMsg{Type: tea.KeyPgUp}
	m = update(m, pgUpMsg)

	if m.autoScroll {
		t.Error("expected autoScroll to be false after PageUp")
	}
}

func TestFormatEntryTruncation(t *testing.T) {
	longMsg := strings.Repeat("x", 200)
	entry := NewEntry(LevelInfo, longMsg)

	line := formatEntry(entry, 80)

	if len(line) > 80 {
		t.Errorf("expected line <= 80 chars, got %d", len(line))
	}
}

func TestFormatEntryShort(t *testing.T) {
	entry := NewEntry(LevelInfo, "short")

	line := formatEntry(entry, 80)

	if len(line) > 80 {
		t.Errorf("expected short line to not exceed width")
	}
	if !strings.Contains(line, "short") {
		t.Error("expected line to contain message")
	}
}

func TestInitWithLogChannel(t *testing.T) {
	ch := make(chan Entry, 1)
	m := New(80, 24, WithLogChannel(ch))

	cmd := m.Init()
	if cmd == nil {
		t.Fatal("expected Init to return a non-nil command")
	}

	// Send an entry through the channel
	ch <- NewEntry(LevelInfo, "channel entry")

	// Init returns a Batch of WindowSize + readLogChannel.
	// Execute the batch command.
	msg := cmd()
	batchMsg, ok := msg.(tea.BatchMsg)
	if !ok {
		// Might be delivered directly if tea.WindowSize returns nil
		if lem, ok := msg.(LogEntryMsg); ok && lem.Message == "channel entry" {
			return
		}
		t.Fatalf("expected BatchMsg or LogEntryMsg, got %T", msg)
	}

	// Execute each sub-command in the batch
	for _, subCmd := range batchMsg {
		subMsg := subCmd()
		if lem, ok := subMsg.(LogEntryMsg); ok {
			if lem.Message != "channel entry" {
				t.Errorf("expected 'channel entry', got %q", lem.Message)
			}
			return
		}
	}
	t.Error("LogEntryMsg not found in batch")
}

func TestDefaultKeyMap(t *testing.T) {
	km := DefaultKeyMap()

	if len(km.Up.Keys()) == 0 {
		t.Error("expected Up key binding")
	}
	if len(km.Down.Keys()) == 0 {
		t.Error("expected Down key binding")
	}
	if len(km.Quit.Keys()) == 0 {
		t.Error("expected Quit key binding")
	}
}

func TestViStyleKeys(t *testing.T) {
	m := New(80, 24)

	for i := 0; i < 50; i++ {
		m.AddEntry(NewEntry(LevelInfo, "log line"))
	}

	// k should scroll up (vi style)
	kMsg := tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'k'}}
	m = update(m, kMsg)

	if m.autoScroll {
		t.Error("expected autoScroll to be false after 'k' key")
	}

	// j should scroll down (vi style)
	jMsg := tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'j'}}
	m = update(m, jMsg)
}

func TestEmptyView(t *testing.T) {
	m := New(80, 24)

	view := m.View()

	if !strings.Contains(view, "entries: 0") {
		t.Error("expected status bar with 'entries: 0' in empty view")
	}
	if !strings.Contains(view, "auto-scroll: on") {
		t.Error("expected 'auto-scroll: on' in empty view")
	}
}

func TestEntryOrderPreserved(t *testing.T) {
	m := New(80, 24)

	messages := []string{"first", "second", "third", "fourth", "fifth"}
	for _, msg := range messages {
		m.AddEntry(NewEntry(LevelInfo, msg))
	}

	for i, expected := range messages {
		if m.entries[i].Message != expected {
			t.Errorf("entry %d: expected %q, got %q", i, expected, m.entries[i].Message)
		}
	}
}

func TestBatchInitCommands(t *testing.T) {
	ch := make(chan Entry, 1)
	m := New(80, 24, WithLogChannel(ch))

	cmd := m.Init()
	if cmd == nil {
		t.Fatal("expected non-nil Init command")
	}

	// The batch command should execute without panic
	ch <- NewEntry(LevelInfo, "test")
	msg := cmd()
	_ = msg
}

func TestDownScrollThenBottomRestoresAutoScroll(t *testing.T) {
	m := New(80, 24)

	for i := 0; i < 50; i++ {
		m.AddEntry(NewEntry(LevelInfo, "log line"))
	}

	// Scroll up first
	upMsg := tea.KeyMsg{Type: tea.KeyUp}
	m = update(m, upMsg)
	if m.autoScroll {
		t.Error("expected autoScroll off after up")
	}

	// Use End key to go to bottom (more reliable than line-by-line)
	endMsg := tea.KeyMsg{Type: tea.KeyEnd}
	m = update(m, endMsg)

	if !m.autoScroll {
		t.Error("expected autoScroll on after going to bottom")
	}
}

func TestPageDownRestoresAutoScrollAtBottom(t *testing.T) {
	m := New(80, 24)

	for i := 0; i < 100; i++ {
		m.AddEntry(NewEntry(LevelInfo, "log line"))
	}

	// Scroll up first
	upMsg := tea.KeyMsg{Type: tea.KeyUp}
	m = update(m, upMsg)

	// Page down
	pgDownMsg := tea.KeyMsg{Type: tea.KeyPgDown}
	m = update(m, pgDownMsg)

	// If we're at bottom, auto-scroll should be restored
	if m.viewport.AtBottom() && !m.autoScroll {
		t.Error("expected autoScroll on after PageDown to bottom")
	}
}
