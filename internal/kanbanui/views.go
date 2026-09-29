package kanbanui

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/key"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/nous-research/hermes-agent/internal/kanbanui/logviewer"
)

type refreshMsg BoardSnapshot

// TaskLogMsg carries a log line from the tailing service.
type TaskLogMsg struct {
	TaskID string
	Entry  logviewer.Entry
}

// Init implements tea.Model.
func (m *Model) Init() tea.Cmd {
	return tea.Batch(
		m.refreshCmd(),
		tea.WindowSize(),
	)
}

// Update implements tea.Model.
func (m *Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		// Log viewer handles resize internally via its own Update.
		if m.currentView == ViewLogs {
			result, cmd := m.logViewer.Update(msg)
			m.logViewer = result.(logviewer.Model)
			_ = cmd
		}
		return m, nil

	case refreshMsg:
		snapshot := BoardSnapshot(msg)
		if snapshot.LoadError != nil {
			m.status = snapshot.LoadError.Error()
		} else {
			m.columns = snapshot.Columns
			m.tasks = snapshot.AllTasks()
			m.status = fmt.Sprintf("loaded %d tasks from %s", len(m.tasks), snapshot.Source)
		}
		return m, nil

	case TaskLogMsg:
		if m.currentView == ViewLogs && msg.TaskID == m.viewingTask {
			m.logViewer.AddEntry(msg.Entry)
		}
		return m, nil

	case tea.KeyMsg:
		// Handle keys differently based on current view
		switch m.currentView {
		case ViewBoard:
			return m.handleBoardKeys(msg)
		case ViewLogs:
			return m.handleLogKeys(msg)
		}
	}

	return m, nil
}

// handleBoardKeys processes keyboard input when the board is visible.
func (m *Model) handleBoardKeys(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch {
	case key.Matches(msg, m.keymap.Quit):
		return m, tea.Quit

	case key.Matches(msg, m.keymap.Refresh):
		return m, m.refreshCmd()

	case key.Matches(msg, m.keymap.Up):
		if m.cursor > 0 {
			m.cursor--
		}
		return m, nil

	case key.Matches(msg, m.keymap.Down):
		if m.cursor < len(m.tasks)-1 {
			m.cursor++
		}
		return m, nil

	case key.Matches(msg, m.keymap.Help):
		return m, nil

	case key.Matches(msg, m.keymap.Log):
		// Open log viewer for the selected task
		return m.openLogViewer()
	}

	return m, nil
}

// handleLogKeys processes keyboard input when the log viewer is visible.
func (m *Model) handleLogKeys(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	// Escape or 'b' returns to board view
	if msg.Type == tea.KeyEsc || (msg.Type == tea.KeyRunes && len(msg.Runes) == 1 && msg.Runes[0] == 'b') {
		return m.closeLogViewer()
	}

	// Delegate to log viewer
	result, lvCmd := m.logViewer.Update(msg)
	m.logViewer = result.(logviewer.Model)
	return m, lvCmd
}

// View implements tea.Model.
func (m *Model) View() string {
	switch m.currentView {
	case ViewLogs:
		return m.renderLogView()
	default:
		return m.render()
	}
}

// openLogViewer switches to the log viewer for the currently selected task.
func (m *Model) openLogViewer() (*Model, tea.Cmd) {
	if len(m.tasks) == 0 || m.cursor >= len(m.tasks) {
		m.status = "no task selected"
		return m, nil
	}

	task := m.tasks[m.cursor]
	m.currentView = ViewLogs
	m.viewingTask = task.ID

	// Create the log viewer with the log channel
	m.logViewer = logviewer.New(m.width, m.height-2, logviewer.WithLogChannel(m.logChan))

	// Seed initial log entries from task events
	m.logViewer.AddEntry(logviewer.NewEntry(
		logviewer.LevelInfo,
		fmt.Sprintf("Task: %s", task.Title),
	))
	m.logViewer.AddEntry(logviewer.NewEntry(
		logviewer.LevelInfo,
		fmt.Sprintf("ID: %s  Status: %s  Assignee: %s", task.ID, task.Status, task.Assignee),
	))
	m.logViewer.AddEntry(logviewer.NewEntry(
		logviewer.LevelDebug,
		"---",
	))

	// Load historical events from the store
	m.loadTaskEvents(task.ID)

	// Start tailing new events
	return m, m.startTailing(task.ID)
}

// closeLogViewer returns to the board view.
func (m *Model) closeLogViewer() (*Model, tea.Cmd) {
	// Stop the tailer
	select {
	case m.tailerDone <- struct{}{}:
	default:
	}
	m.currentView = ViewBoard
	m.viewingTask = ""
	return m, nil
}

// startTailing returns a cmd that periodically checks for new task events.
func (m *Model) startTailing(taskID string) tea.Cmd {
	go m.runTailer(taskID)
	return func() tea.Msg {
		return TaskLogMsg{
			TaskID: taskID,
			Entry: logviewer.NewEntry(logviewer.LevelDebug, "tailing started"),
		}
	}
}

// runTailer is a goroutine that polls the DB for new task events and sends them as log entries.
func (m *Model) runTailer(taskID string) {
	lastCheck := time.Now()
	for {
		select {
		case <-m.tailerDone:
			return
		default:
		}

		// Check for new events
		events := m.store.LoadTaskEvents(taskID, lastCheck)
		for _, ev := range events {
			level := logviewer.LevelInfo
			if ev.Kind == "error" || ev.Kind == "crashed" {
				level = logviewer.LevelError
			} else if ev.Kind == "heartbeat" {
				level = logviewer.LevelDebug
			}

			entry := logviewer.NewEntry(level, ev.Message())
			select {
			case m.logChan <- entry:
			default:
				// channel full, skip
			}
			if ev.CreatedAt.After(lastCheck) {
				lastCheck = ev.CreatedAt
			}
		}

		time.Sleep(1 * time.Second)
	}
}

// loadTaskEvents loads historical events for a task and adds them to the log viewer.
func (m *Model) loadTaskEvents(taskID string) {
	events := m.store.LoadTaskEvents(taskID, time.Time{})
	for _, ev := range events {
		level := logviewer.LevelInfo
		if ev.Kind == "error" || ev.Kind == "crashed" {
			level = logviewer.LevelError
		} else if ev.Kind == "heartbeat" {
			level = logviewer.LevelDebug
		}
		m.logViewer.AddEntry(logviewer.NewEntry(level, ev.Message()))
	}
}

func (m *Model) refreshCmd() tea.Cmd {
	return func() tea.Msg {
		return refreshMsg(m.store.Load())
	}
}

// renderLogView renders the log viewer with a header showing the task info.
func (m *Model) renderLogView() string {
	var sb strings.Builder

	// Header
	var taskTitle string
	if m.viewingTask != "" {
		for _, t := range m.tasks {
			if t.ID == m.viewingTask {
				taskTitle = t.Title
				break
			}
		}
	}
	header := fmt.Sprintf("Logs: %s [%s]", m.viewingTask, taskTitle)
	sb.WriteString(m.styles.Header.Render(header))
	sb.WriteString("\n")

	// Log viewer content
	sb.WriteString(m.logViewer.View())
	sb.WriteString("\n")

	// Footer
	sb.WriteString(m.styles.Muted.Render("esc/b: back to board  ↑/↓ scroll  g/G: top/bottom"))

	return sb.String()
}

func formatRefreshInterval(d time.Duration) string {
	if d <= 0 {
		return "manual"
	}
	return d.String()
}

func statusStyle(styles Styles, status TaskStatus) lipgloss.Style {
	if style, ok := styles.Status[string(status)]; ok {
		return style
	}
	return styles.Muted
}

func truncate(text string, width int) string {
	if width <= 0 || len(text) <= width {
		return text
	}
	if width <= 1 {
		return text[:width]
	}
	return strings.TrimSpace(text[:width-1]) + "\u2026"
}
