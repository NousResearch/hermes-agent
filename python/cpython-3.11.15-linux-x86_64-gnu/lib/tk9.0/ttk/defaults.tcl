#
# Settings for default theme.
#

namespace eval ttk::theme::default {

    variable colors
    array set colors {
	-frame			"#d9d9d9"
	-foreground		"#000000"
	-window			"#ffffff"
	-alternate		"#e8e8e8"
	-text			"#000000"
	-activebg		"#ececec"
	-selectbg		"#4a6984"
	-selectfg		"#ffffff"
	-darker			"#c3c3c3"
	-disabledfg		"#a3a3a3"
	-indicator		"#4a6984"
	-disabledindicator	"#a3a3a3"
	-pressedindicator	"#5895bc"
    }

    # On X11, if the user specifies their own choice of colour scheme via
    # X resources, then set the colour palette based on the user's choice.
    if {[tk windowingsystem] eq "x11"} {
	foreach \
		xResourceName {
		    {	background		Background		}
		    {	foreground		Foreground		}
		    {	background		Background		}
		    {	background		Background		}
		    {	foreground		Foreground		}
		    {	activeBackground	ActiveBackground	}
		    {	selectBackground	SelectBackground	}
		    {	selectForeground	SelectForeground	}
		    {	troughColor		TroughColor		}
		    {	disabledForeground	DisabledForeground	}
		    {	selectBackground	SelectBackground	}
		    {	disabledForeground	DisabledForeground	}
		    {	selectBackground	SelectBackground	}
		    {	windowColor		Background		} } \
		colorName {
		    -frame -foreground -window -alternate -text
		    -activebg -selectbg -selectfg
		    -darker -disabledfg -indicator
		    -disabledindicator -pressedindicator -window } {
	    set color [eval option get . $xResourceName]
	    if {$color ne ""} {
		set colors($colorName) $color
	    }
	}
    }

    # This array is used to match up the tk widget options with
    # the corresponding values in the 'colors' array.
    # This is used by tk_setPalette to apply the new palette
    # to the ttk widgets.
    variable colorOptionLookup
    array set colorOptionLookup {
	background		{-frame -window -alternate}
	foreground		{-foreground -text}
	activeBackground	-activebg
	selectBackground	{-selectbg -indicator -pressedindicator}
	selectForeground	-selectfg
	troughColor		-darker
	disabledForeground	{-disabledfg -disabledindicator}
    }
}

# ttk::theme::default::reconfigureDefaultTheme --
#
# This procedure contains the definition of the 'default' theme itself.
# The theme definition is in a procedure, so it can be re-called when
# required, enabling tk_setPalette to set the colours of the ttk widgets.
#
# Arguments:
# None.

proc ttk::theme::default::reconfigureDefaultTheme {} {
    upvar ttk::theme::default::colors colors

    # The definition of the 'default' theme.

    ttk::style theme settings default {

	ttk::style configure "." \
	    -borderwidth	1 \
	    -background		$colors(-frame) \
	    -foreground		$colors(-foreground) \
	    -troughcolor	$colors(-darker) \
	    -font		TkDefaultFont \
	    -selectbackground	$colors(-selectbg) \
	    -selectforeground	$colors(-selectfg) \
	    -insertwidth	1 \
	    -insertcolor	$colors(-foreground) \
	    -focuscolor		$colors(-text)

	ttk::style map "." -background \
	    [list disabled $colors(-frame)  active $colors(-activebg)]
	ttk::style map "." -foreground \
	    [list disabled $colors(-disabledfg)]

	ttk::style configure TButton \
	    -anchor center -padding 2.25p -width -9 \
	    -relief raised -shiftrelief 1
	ttk::style map TButton -relief [list {!disabled pressed} sunken]

	foreach style {TCheckbutton TRadiobutton} {
	    ttk::style configure $style \
		-indicatorbackground $colors(-window) \
		-indicatorforeground $colors(-selectfg) \
		-indicatormargin {0 1.5p 3p 1.5p} -padding 0.75p
	    ttk::style map $style -indicatorbackground \
		[list {alternate disabled}	$colors(-disabledindicator) \
		      {alternate pressed}	$colors(-pressedindicator) \
		      alternate			$colors(-indicator) \
		      {selected disabled}	$colors(-disabledindicator) \
		      {selected pressed}	$colors(-pressedindicator) \
		      selected			$colors(-indicator) \
		      disabled			$colors(-frame) \
		      pressed			$colors(-darker)]
	}

	ttk::style configure TMenubutton \
	    -relief raised -arrowsize 3.75p -arrowpadding 2.25p \
	    -arrowcolor $colors(-text) -padding {7.5p 2.25p}
	ttk::style map TMenubutton \
	    -arrowcolor [list disabled $colors(-disabledfg)]

	ttk::style configure TEntry \
	    -fieldbackground $colors(-window) -padding 1 \
	    -focuswidth 2 -focuscolor $colors(-selectbg)
	ttk::style map TEntry -fieldbackground \
	    [list readonly $colors(-frame) disabled $colors(-frame)]

	ttk::style configure TCombobox \
	    -arrowsize 9p -arrowcolor $colors(-text) \
	    -fieldbackground $colors(-window) -padding 1 \
	    -focuswidth 1 -focuscolor $colors(-selectbg)
	ttk::style map TCombobox -fieldbackground \
	    [list readonly $colors(-frame) disabled $colors(-frame)] \
	    -arrowcolor [list disabled $colors(-disabledfg)]

	ttk::style configure TSpinbox \
	    -arrowsize 7.5p -arrowcolor $colors(-text) \
	    -fieldbackground $colors(-window) -padding {1.5p 0 7.5p 0} \
	    -focuswidth 1 -focuscolor $colors(-selectbg)
	ttk::style map TSpinbox -fieldbackground \
	    [list readonly $colors(-frame) disabled $colors(-frame)] \
	    -arrowcolor [list disabled $colors(-disabledfg)]

	ttk::style configure TLabelframe \
	    -relief groove -borderwidth 2

	ttk::style configure TScrollbar \
	    -width 9p -arrowsize 9p -arrowcolor $colors(-text)
	ttk::style map TScrollbar \
	    -arrowcolor [list disabled $colors(-disabledfg)]

	ttk::style configure TScale \
	    -innercolor $colors(-selectbg) \
	    -outercolor $colors(-window) \
	    -bordercolor $colors(-darker) \
	    -groovewidth 3p
	ttk::style map TScale \
	    -outercolor [list active $colors(-activebg)]

	ttk::style configure TProgressbar \
	    -background $colors(-selectbg) \
	    -borderwidth 0 \
	    -barsize 22.5p \
	    -thickness 3p

	ttk::style configure TNotebook.Tab \
	    -padding {3p 1.5p} -background $colors(-darker)
	ttk::style map TNotebook.Tab \
	    -background [list selected $colors(-frame)] \
	    -highlight [list selected 1] \
	    -highlightcolor [list selected $colors(-selectbg)]

	# Treeview.
	#
	ttk::style configure Heading -font TkHeadingFont -relief raised
	ttk::style configure Item -indicatorsize 9p \
	    -indicatormargins {1.5p 1.5p 3p 1.5p}
	ttk::style configure Treeview \
	    -background $colors(-window) \
	    -stripedbackground $colors(-alternate) \
	    -fieldbackground $colors(-window) \
	    -foreground $colors(-text) \
	    -indent 15p \
	    -focuswidth 1 -focuscolor $colors(-selectbg)
	ttk::setTreeviewRowHeight
	ttk::style configure Treeview.Separator \
	    -background $colors(-alternate)
	ttk::style map Treeview \
	    -background [list disabled $colors(-frame)\
				selected $colors(-selectbg)] \
	    -foreground [list disabled $colors(-disabledfg) \
				selected $colors(-selectfg)]

	# Combobox popdown frame
	ttk::style layout ComboboxPopdownFrame {
	    ComboboxPopdownFrame.border -sticky nswe
	}
	ttk::style configure ComboboxPopdownFrame \
	    -borderwidth 1 -relief solid

	#
	# Toolbar buttons:
	#
	ttk::style layout Toolbutton {
	    Toolbutton.border -children {
		Toolbutton.focus -children {
		    Toolbutton.padding -children {
			Toolbutton.label
		    }
		}
	    }
	}
	ttk::style configure Toolbutton \
	    -padding 1.5p -relief flat
	ttk::style map Toolbutton -relief \
	    [list disabled flat selected sunken pressed sunken active raised]
	ttk::style map Toolbutton -background \
	    [list pressed $colors(-darker)  active $colors(-activebg)]
    }
}

ttk::theme::default::reconfigureDefaultTheme
