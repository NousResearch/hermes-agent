#
# Settings for 'winnative' theme.
#

namespace eval ttk::theme::winnative {

    ttk::style theme settings winnative {

	ttk::style configure "." \
	    -background SystemButtonFace \
	    -foreground SystemWindowText \
	    -selectforeground SystemHighlightText \
	    -selectbackground SystemHighlight \
	    -fieldbackground SystemWindow \
	    -insertcolor SystemWindowText \
	    -troughcolor SystemScrollbar \
	    -font TkDefaultFont

	ttk::style map "." -foreground {disabled SystemGrayText}
	ttk::style map "." -embossed {disabled 1}

	ttk::style configure TButton \
	    -anchor center -width -11 -relief raised -shiftrelief 1
	ttk::style map TButton -relief {{!disabled pressed} sunken}

	ttk::style configure TCheckbutton -padding {1.5p 3p}
	ttk::style configure TRadiobutton -padding {1.5p 3p}

	ttk::style configure TMenubutton \
	    -padding {6p 3p} -arrowsize 2.25p -relief raised

	ttk::style configure TEntry \
	    -padding 2 -insertwidth 1
	ttk::style map TEntry \
	    -fieldbackground \
		{readonly SystemButtonFace disabled SystemButtonFace} \
	    -selectbackground {!focus SystemWindow} \
	    -selectforeground {!focus SystemWindowText}

	ttk::style configure TCombobox -padding 1.5p
	ttk::style map TCombobox \
	    -selectbackground [list !focus SystemWindow] \
	    -selectforeground [list !focus SystemWindowText] \
	    -fieldbackground [list \
		readonly SystemButtonFace \
		disabled SystemButtonFace] \
	    -foreground	[list \
		disabled		SystemGrayText \
		{readonly focus}	SystemHighlightText \
	    ] \
	    -focusfill {{readonly focus} SystemHighlight}

	ttk::style element create ComboboxPopdownFrame.border from default
	ttk::style configure ComboboxPopdownFrame \
	    -borderwidth 1 -relief solid

	ttk::style configure TSpinbox -padding {1.5p 0 12p 0}

	ttk::style configure TLabelframe -borderwidth 2 -relief groove

	ttk::style configure Toolbutton -relief flat -padding {6p 3p}
	ttk::style map Toolbutton -relief \
	    {disabled flat  selected sunken  pressed sunken  active raised}

	ttk::style configure TScale -groovewidth 3p

	ttk::style configure TNotebook -tabmargins {2 2 2 0}
	ttk::style configure TNotebook.Tab -padding {2.25p 0.75p} -borderwidth 1
	ttk::style map TNotebook.Tab -expand {selected {2 2 2 0}}

	# Treeview:
	ttk::style configure Heading -font TkHeadingFont -relief raised
	ttk::style configure Item \
	    -indicatormargins {1.5p 1.5p 3p 1.5p}
	ttk::style configure Treeview -background SystemWindow \
	    -stripedbackground System3dLight -indent 15p
	ttk::setTreeviewRowHeight
	ttk::style map Treeview \
	    -background [list   disabled SystemButtonFace \
				selected SystemHighlight] \
	    -foreground [list   disabled SystemGrayText \
				selected SystemHighlightText]

	ttk::style configure TProgressbar \
	    -background SystemHighlight -borderwidth 0 \
	    -barsize 22.5p -thickness 11.25p
    }
}

# ttk::theme::winnative::configureNotebookStyle --
#
# Sets theme-specific option values for the ttk::notebook style $style and the
# style $style.Tab.  Invoked by ::ttk::configureNotebookStyle.

proc ttk::theme::winnative::configureNotebookStyle {style} {
    set tabPos [ttk::style lookup $style -tabposition {} nw]
    switch -- [string index $tabPos 0] {
	n {
	    ttk::style configure $style -tabmargins     {2 2 2 0}
	    ttk::style map $style.Tab -expand {selected {2 2 2 0}}
	}
	s {
	    ttk::style configure $style -tabmargins     {2 0 2 2}
	    ttk::style map $style.Tab -expand {selected {2 0 2 2}}
	}
	w {
	    ttk::style configure $style -tabmargins     {2 2 0 2}
	    ttk::style map $style.Tab -expand {selected {2 2 0 2}}
	}
	e {
	    ttk::style configure $style -tabmargins     {0 2 2 2}
	    ttk::style map $style.Tab -expand {selected {0 2 2 2}}
	}
	default {
	    ttk::style configure $style -tabmargins     {2 2 2 0}
	    ttk::style map $style.Tab -expand {selected {2 2 2 0}}
	}
    }
}
