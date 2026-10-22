package kanbanui

import "github.com/charmbracelet/lipgloss"

// Styles contains Lip Gloss styles used by the placeholder renderer.
type Styles struct {
	Header   lipgloss.Style
	Muted    lipgloss.Style
	Selected lipgloss.Style
	Status   map[string]lipgloss.Style
}

// DefaultStyles returns the initial dark-friendly style palette.
func DefaultStyles() Styles {
	return Styles{
		Header: lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("#88C0D0")),
		Muted:  lipgloss.NewStyle().Foreground(lipgloss.Color("#787878")),
		Selected: lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#1E1E2E")).
			Background(lipgloss.Color("#88C0D0")),
		Status: map[string]lipgloss.Style{
			"triage":  lipgloss.NewStyle().Foreground(lipgloss.Color("#D08770")),
			"todo":    lipgloss.NewStyle().Foreground(lipgloss.Color("#81A1C1")),
			"ready":   lipgloss.NewStyle().Foreground(lipgloss.Color("#A3BE8C")),
			"running": lipgloss.NewStyle().Foreground(lipgloss.Color("#EBCB8B")),
			"blocked": lipgloss.NewStyle().Foreground(lipgloss.Color("#BF616A")),
			"done":    lipgloss.NewStyle().Foreground(lipgloss.Color("#8FBCBB")),
		},
	}
}
