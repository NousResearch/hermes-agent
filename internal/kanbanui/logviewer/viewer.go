package logviewer

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
)

// KeyMap defines the keyboard bindings for the log viewer.
type KeyMap struct {
	Up       key.Binding
	Down     key.Binding
	PageUp   key.Binding
	PageDown key.Binding
	Home     key.Binding
	End      key.Binding
	Quit     key.Binding
}

// DefaultKeyMap returns the standard key bindings.
func DefaultKeyMap() KeyMap {
	return KeyMap{
		Up: key.NewBinding(
			key.WithKeys("up", "k"),
			key.WithHelp("up/k", "scroll up"),
		),
		Down: key.NewBinding(
			key.WithKeys("down", "j"),
			key.WithHelp("down/j", "scroll down"),
		),
		PageUp: key.NewBinding(
			key.WithKeys("pgup", "b", "ctrl+b"),
			key.WithHelp("pgup/b", "page up"),
		),
		PageDown: key.NewBinding(
			key.WithKeys("pgdown", "f", "ctrl+f"),
			key.WithHelp("pgdn/f", "page down"),
		),
		Home: key.NewBinding(
			key.WithKeys("home", "g"),
			key.WithHelp("home/g", "go to start"),
		),
		End: key.NewBinding(
			key.WithKeys("end", "G", "shift+G"),
			key.WithHelp("end/G", "go to end"),
		),
		Quit: key.NewBinding(
			key.WithKeys("q", "ctrl+c"),
			key.WithHelp("q/ctrl+c", "quit"),
		),
	}
}

// LogEntryMsg is delivered to the viewer when a new log entry arrives.
type LogEntryMsg struct {
	Entry
}

// Model is the Bubbletea model for the log viewer.
type Model struct {
	entries    []Entry
	viewport   viewport.Model
	width      int
	height     int
	keyMap     KeyMap
	autoScroll bool
	logChan    chan Entry // optional: if set, goroutine reads from it
}

// Options configure the Model.
type Option func(*Model)

// WithKeyMap sets a custom key map.
func WithKeyMap(km KeyMap) Option {
	return func(m *Model) { m.keyMap = km }
}

// WithLogChannel sets a channel to read log entries from automatically.
// The model will spawn a goroutine that forwards channel receives as LogEntryMsg.
func WithLogChannel(ch chan Entry) Option {
	return func(m *Model) { m.logChan = ch }
}

// New creates a new log viewer model.
func New(width, height int, opts ...Option) Model {
	vp := viewport.New(width, height)

	m := Model{
		entries:    make([]Entry, 0),
		viewport:   vp,
		width:      width,
		height:     height,
		keyMap:     DefaultKeyMap(),
		autoScroll: true,
	}

	for _, opt := range opts {
		opt(&m)
	}

	return m
}

// Init initializes the Tea model.
func (m Model) Init() tea.Cmd {
	var cmds []tea.Cmd
	cmds = append(cmds, tea.WindowSize())

	if m.logChan != nil {
		cmds = append(cmds, readLogChannel(m.logChan))
	}

	return tea.Batch(cmds...)
}

// readLogChannel is a command that reads from the log channel and sends messages.
func readLogChannel(ch chan Entry) tea.Cmd {
	return func() tea.Msg {
		entry := <-ch
		return LogEntryMsg{Entry: entry}
	}
}

// AddEntry appends a log entry and re-renders the content.
func (m *Model) AddEntry(entry Entry) {
	m.entries = append(m.entries, entry)
	m.autoScroll = m.viewport.AtBottom()
	m.updateViewportContent()
	if m.autoScroll {
		m.viewport.GotoBottom()
	}
}

// updateViewportContent rebuilds the viewport content from entries.
func (m *Model) updateViewportContent() {
	var sb strings.Builder
	entryWidth := m.viewport.Width
	for _, e := range m.entries {
		line := formatEntry(e, entryWidth)
		sb.WriteString(line)
		sb.WriteString("\n")
	}
	m.viewport.SetContent(sb.String())
}

// formatEntry formats a single log entry as a styled text line.
func formatEntry(e Entry, width int) string {
	timeStr := e.Time.Format("15:04:05.000")
	levelStr := e.Level.String()
	line := fmt.Sprintf("[%s] %s | %s", timeStr, levelStr, e.Message)
	if len(line) > width {
		return line[:width]
	}
	return line
}

// Update handles Tea messages.
func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd

	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch {
		case key.Matches(msg, m.keyMap.Quit):
			return m, tea.Quit
		case key.Matches(msg, m.keyMap.Up):
			m.viewport.LineUp(1)
			m.autoScroll = false
		case key.Matches(msg, m.keyMap.Down):
			m.viewport.LineDown(1)
			if m.viewport.AtBottom() {
				m.autoScroll = true
			}
		case key.Matches(msg, m.keyMap.PageUp):
			m.viewport.HalfViewUp()
			m.autoScroll = false
		case key.Matches(msg, m.keyMap.PageDown):
			m.viewport.HalfViewDown()
			if m.viewport.AtBottom() {
				m.autoScroll = true
			}
		case key.Matches(msg, m.keyMap.Home):
			m.viewport.GotoTop()
			m.autoScroll = false
		case key.Matches(msg, m.keyMap.End):
			m.viewport.GotoBottom()
			m.autoScroll = true
		}

	case LogEntryMsg:
		m.AddEntry(msg.Entry)

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.viewport.Width = msg.Width
		m.viewport.Height = msg.Height
		m.updateViewportContent()
		// If auto-scroll was on before resize, stay at bottom.
		if m.autoScroll {
			m.viewport.GotoBottom()
		}
	}

	m.viewport, cmd = m.viewport.Update(msg)
	return m, cmd
}

// View renders the log viewer.
func (m Model) View() string {
	var sb strings.Builder
	sb.WriteString(m.viewport.View())
	sb.WriteString("\n")

	// Status bar
	var parts []string
	parts = append(parts, fmt.Sprintf("entries: %d", len(m.entries)))
	if m.autoScroll {
		parts = append(parts, "auto-scroll: on")
	} else {
		parts = append(parts, "auto-scroll: off")
	}
	parts = append(parts, "q:quit")

	bar := strings.Join(parts, "  ")
	sb.WriteString(bar)
	return sb.String()
}
