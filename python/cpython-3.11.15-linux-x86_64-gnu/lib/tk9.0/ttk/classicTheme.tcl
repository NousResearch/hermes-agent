#
# "classic" Tk theme.
#
# Implements Tk's traditional Motif-like look and feel.
#

namespace eval ttk::theme::classic {

    variable colors
    array set colors {
	-frame		"#d9d9d9"
	-window		"#ffffff"
	-alternate	"#f0f0f0"
	-activebg	"#ececec"
	-troughbg	"#b3b3b3"
	-selectbg	"#c3c3c3"
	-selectfg	"#000000"
	-disabledfg	"#a3a3a3"
	-indicator	"#b03060"
	-altindicator	"#b05e5e"
    }

    ttk::style theme settings classic {
	ttk::style configure "." \
	    -font		TkDefaultFont \
	    -background		$colors(-frame) \
	    -foreground		black \
	    -selectbackground	$colors(-selectbg) \
	    -selectforeground	$colors(-selectfg) \
	    -troughcolor	$colors(-troughbg) \
	    -indicatorcolor	$colors(-frame) \
	    -highlightcolor	$colors(-frame) \
	    -highlightthickness	1 \
	    -borderwidth	1 \
	    -insertwidth	2 \
	    -focuswidth		0

	# To match pre-Xft X11 appearance, use:
	#	ttk::style configure . -font {Helvetica 12 bold}

	ttk::style map "." -background \
	    [list disabled $colors(-frame) active $colors(-activebg)]
	ttk::style map "." -foreground \
	    [list disabled $colors(-disabledfg)]

	ttk::style map "." -highlightcolor {focus black}

	ttk::style configure TButton \
	    -anchor center -padding "3m 1m" -relief raised -shiftrelief 1
	ttk::style map TButton -relief {{!disabled pressed} sunken}

	ttk::style configure TCheckbutton -indicatorrelief raised \
	    -indicatormargin {0 1.5p 3p 1.5p}
	ttk::style map TCheckbutton \
	    -indicatorcolor [list \
		    pressed $colors(-frame) \
		    alternate $colors(-altindicator) \
		    selected $colors(-indicator)] \
	    -indicatorrelief {alternate raised  selected sunken  pressed sunken}

	ttk::style configure TRadiobutton -indicatorrelief raised \
	    -indicatormargin {0 1.5p 3p 1.5p}
	ttk::style map TRadiobutton \
	    -indicatorcolor [list \
		    pressed $colors(-frame) \
		    alternate $colors(-altindicator) \
		    selected $colors(-indicator)] \
	    -indicatorrelief {alternate raised  selected sunken  pressed sunken}

	ttk::style configure TMenubutton -relief raised \
	    -indicatormargin {3.75p 0} -padding {3m 1m}

	ttk::style configure TEntry -padding 1 -font TkTextFont
	ttk::style map TEntry -fieldbackground \
		[list readonly $colors(-frame) disabled $colors(-frame)]

	ttk::style element create Combobox.downarrow from default
	ttk::style configure TCombobox -padding 1 -arrowsize 9p
	ttk::style map TCombobox -fieldbackground \
		[list readonly $colors(-frame) disabled $colors(-frame)]
	ttk::style configure ComboboxPopdownFrame \
	    -relief solid -borderwidth 1

	ttk::style element create Spinbox.uparrow from default
	ttk::style element create Spinbox.downarrow from default
	ttk::style configure TSpinbox -arrowsize 7.5p -padding {1.5p 0 7.5p 0}
	ttk::style map TSpinbox -fieldbackground \
	    [list readonly $colors(-frame) disabled $colors(-frame)]

	ttk::style configure TLabelframe -borderwidth 2 -relief groove

	ttk::style configure TScrollbar -relief raised -arrowsize 9p -width 9p
	ttk::style map TScrollbar -relief {{pressed !disabled} sunken}

	ttk::style configure TScale -sliderrelief raised \
	    -sliderlength 22.5p -sliderthickness 11.25p
	ttk::style map TScale -sliderrelief {{pressed !disabled} sunken}

	ttk::style configure TProgressbar -background SteelBlue \
	    -barsize 22.5p -thickness 11.25p

	ttk::style configure TNotebook.Tab \
	    -padding {3m 1m} \
	    -background $colors(-troughbg) \
	    -focussolid 1
	ttk::style map TNotebook.Tab -background [list selected $colors(-frame)]

	# Treeview:
	ttk::style configure Heading -font TkHeadingFont -relief raised
	ttk::style configure Item -indicatorsize 9p \
	    -indicatormargins {1.5p 1.5p 3p 1.5p}
	ttk::style configure Treeview -background $colors(-window) \
	    -stripedbackground $colors(-alternate) -indent 15p
	ttk::setTreeviewRowHeight
	ttk::style configure Treeview.Separator \
	    -background $colors(-alternate)
	ttk::style map Treeview \
	    -background [list disabled $colors(-frame)\
				selected $colors(-selectbg)] \
	    -foreground [list disabled $colors(-disabledfg) \
				selected $colors(-selectfg)]

	#
	# Toolbar buttons:
	#
	ttk::style layout Toolbutton {
	    Toolbutton.focus -children {
		Toolbutton.border -children {
		    Toolbutton.padding -children {
			Toolbutton.label
		    }
		}
	    }
	}
	ttk::style configure Toolbutton -padding 1.5p -relief flat \
	    -shiftrelief 2 -focussolid 1
	ttk::style map Toolbutton -relief \
	    {disabled flat selected sunken pressed sunken active raised}
	ttk::style map Toolbutton -background \
	    [list pressed $colors(-troughbg)  active $colors(-activebg)]

	ttk::style configure Sash \
	    -sashthickness 4.5p -sashpad 1.5 -handlesize 6p -handlepad 6p
    }
}
