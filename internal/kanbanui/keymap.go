package kanbanui

import "github.com/charmbracelet/bubbles/key"

// KeyMap groups keyboard bindings for the kanban TUI.
type KeyMap struct {
	Up      key.Binding
	Down    key.Binding
	Refresh key.Binding
	Help    key.Binding
	Quit    key.Binding
	Log     key.Binding
}

// DefaultKeyMap returns the initial navigation bindings.
func DefaultKeyMap() KeyMap {
	return KeyMap{
		Up:      key.NewBinding(key.WithKeys("up", "k"), key.WithHelp("↑/k", "up")),
		Down:    key.NewBinding(key.WithKeys("down", "j"), key.WithHelp("↓/j", "down")),
		Refresh: key.NewBinding(key.WithKeys("ctrl+r"), key.WithHelp("C-r", "refresh")),
		Help:    key.NewBinding(key.WithKeys("?"), key.WithHelp("?", "help")),
		Quit:    key.NewBinding(key.WithKeys("q", "ctrl+c"), key.WithHelp("q", "quit")),
		Log:     key.NewBinding(key.WithKeys("l"), key.WithHelp("l", "logs")),
	}
}

// ShortHelp implements help.KeyMap.
func (k KeyMap) ShortHelp() []key.Binding {
	return []key.Binding{k.Up, k.Down, k.Log, k.Refresh, k.Quit}
}

// FullHelp implements help.KeyMap.
func (k KeyMap) FullHelp() [][]key.Binding {
	return [][]key.Binding{k.ShortHelp()}
}
