#
# "Clam" theme.
#
# Inspired by the XFCE family of Gnome themes.
#

namespace eval ttk::theme::clam {

    variable colors
    array set colors {
	-disabledfg		"#999999"
	-frame			"#dcdad5"
	-window			"#ffffff"
	-dark			"#cfcdc8"
	-darker			"#bab5ab"
	-darkest		"#9e9a91"
	-lighter		"#eeebe7"
	-lightest		"#ffffff"
	-selectbg		"#4a6984"
	-selectfg		"#ffffff"
	-altindicator		"#5895bc"
	-disabledaltindicator	"#a0a0a0"
    }

    ttk::style theme settings clam {

	ttk::style configure "." \
	    -background $colors(-frame) \
	    -foreground black \
	    -bordercolor $colors(-darkest) \
	    -darkcolor $colors(-dark) \
	    -lightcolor $colors(-lighter) \
	    -troughcolor $colors(-darker) \
	    -selectbackground $colors(-selectbg) \
	    -selectforeground $colors(-selectfg) \
	    -font TkDefaultFont

	ttk::style map "." \
	    -background [list disabled $colors(-frame) \
			     active $colors(-lighter)] \
	    -foreground [list disabled $colors(-disabledfg)] \
	    -selectbackground [list !focus $colors(-darkest)] \
	    -selectforeground [list !focus white]

	# -selectbackground [list  !focus "#847d73"]

	ttk::style configure TButton \
	    -anchor center -width -11 -padding 3.75p -relief raised
	ttk::style map TButton \
	    -background [list \
			     disabled $colors(-frame) \
			     pressed $colors(-darker) \
			     active $colors(-lighter)] \
	    -lightcolor [list pressed $colors(-darker)] \
	    -darkcolor [list pressed $colors(-darker)] \
	    -bordercolor {alternate #000000}

	ttk::style configure Toolbutton \
	    -anchor center -padding 1.5p -relief flat
	ttk::style map Toolbutton \
	    -relief [list \
		    disabled flat \
		    selected sunken \
		    pressed sunken \
		    active raised] \
	    -background [list \
		    disabled $colors(-frame) \
		    pressed $colors(-darker) \
		    active $colors(-lighter)] \
	    -lightcolor [list pressed $colors(-darker)] \
	    -darkcolor [list pressed $colors(-darker)]

	ttk::style configure TCheckbutton \
	    -indicatorbackground "#ffffff" \
	    -indicatorforeground "#000000" \
	    -indicatormargin {0.75p 0.75p 3p 0.75p} \
	    -padding 1.5p
	ttk::style configure TRadiobutton \
	    -indicatorbackground "#ffffff" \
	    -indicatorforeground "#000000" \
	    -indicatormargin {0.75p 0.75p 3p 0.75p} \
	    -padding 1.5p
	ttk::style map TCheckbutton \
	    -indicatorbackground [list \
		    pressed		  $colors(-frame) \
		    {alternate disabled}  $colors(-disabledaltindicator) \
		    alternate		  $colors(-altindicator) \
		    disabled		  $colors(-frame)] \
	    -indicatorforeground [list disabled $colors(-disabledfg)]
	ttk::style map TRadiobutton \
	    -indicatorbackground [list \
		    pressed		  $colors(-frame) \
		    {alternate disabled}  $colors(-disabledaltindicator) \
		    alternate		  $colors(-altindicator) \
		    disabled		  $colors(-frame)] \
	    -indicatorforeground [list disabled $colors(-disabledfg)]

	ttk::style configure TMenubutton \
	    -width -11 -arrowsize 3.75p -arrowpadding 2.25p -padding 3.75p \
	    -relief raised

	ttk::style configure TEntry -padding 1 -insertwidth 1
	ttk::style map TEntry \
	    -background [list readonly $colors(-frame)] \
	    -bordercolor [list focus $colors(-selectbg)] \
	    -lightcolor [list focus #6f9dc6]

	ttk::style configure TCombobox -padding 1 -insertwidth 1 \
	    -arrowsize 10.5p
	ttk::style map TCombobox \
	    -background [list active $colors(-lighter) \
			     pressed $colors(-lighter)] \
	    -fieldbackground [list {readonly focus} $colors(-selectbg) \
				  readonly $colors(-frame)] \
	    -foreground [list {readonly focus} $colors(-selectfg)] \
	    -arrowcolor [list disabled $colors(-disabledfg)] \
	    -bordercolor [list focus $colors(-selectbg)]
	ttk::style configure ComboboxPopdownFrame \
	    -relief solid -borderwidth 1

	ttk::style configure TSpinbox -arrowsize 7.5p -padding {1.5p 0 7.5p 0}
	ttk::style map TSpinbox \
	    -background [list readonly $colors(-frame)] \
	    -arrowcolor [list disabled $colors(-disabledfg)] \
	    -bordercolor [list focus $colors(-selectbg)]

	ttk::style configure TNotebook.Tab -padding {4.5p 1.5p 4.5p 1.5p}
	ttk::style map TNotebook.Tab \
	    -padding {selected {4.5p 3p 4.5p 1.5p}} \
	    -background [list selected $colors(-frame) {} $colors(-darker)] \
	    -lightcolor [list selected $colors(-lighter) {} $colors(-dark)]

	# Treeview:
	ttk::style configure Heading \
	    -font TkHeadingFont -relief raised -padding 2.25p
	ttk::style configure Item -indicatorsize 9p \
	    -indicatormargins {1.5p 1.5p 3p 1.5p}
	ttk::style configure Treeview -background $colors(-window) \
	    -stripedbackground $colors(-lighter) -indent 15p
	ttk::setTreeviewRowHeight
	ttk::style configure Treeview.Separator \
	    -background $colors(-lighter)
	ttk::style map Treeview \
	    -background [list disabled $colors(-frame)\
				selected $colors(-selectbg)] \
	    -foreground [list disabled $colors(-disabledfg) \
				selected $colors(-selectfg)] \
	    -bordercolor [list focus $colors(-selectbg)]

	ttk::style configure TLabelframe \
	    -labeloutside true -labelmargins {0 0 0 3p} \
	    -borderwidth 2 -relief raised

	ttk::style configure TScrollbar -gripsize 7.5p \
	    -arrowsize 10.5p -width 10.5p

	ttk::style configure TScale -gripsize 7.5p \
	    -arrowsize 10.5p -sliderlength 22.5p

	ttk::style configure TProgressbar -background $colors(-frame) \
	    -arrowsize 10.5p -sliderlength 22.5p

	ttk::style configure Sash -sashthickness 4.5p -gripsize 15p
    }
}

# ttk::theme::clam::configureNotebookStyle --
#
# Sets theme-specific option values for the ttk::notebook tab style $style.Tab.
# Invoked by ::ttk::configureNotebookStyle.

proc ttk::theme::clam::configureNotebookStyle {style} {
    set tabPos [ttk::style lookup $style -tabposition {} nw]
    switch -- [string index $tabPos 0] {
	n {
	    ttk::style configure $style.Tab -padding     {4.5p 1.5p 4.5p 1.5p}
	    ttk::style map $style.Tab -padding {selected {4.5p 3p   4.5p 1.5p}}
	}
	s {
	    ttk::style configure $style.Tab -padding     {4.5p 1.5p 4.5p 1.5p}
	    ttk::style map $style.Tab -padding {selected {4.5p 1.5p 4.5p 3p  }}
	}
	w {
	    ttk::style configure $style.Tab -padding     {1.5p 4.5p 1.5p 4.5p}
	    ttk::style map $style.Tab -padding {selected {3p   4.5p 1.5p 4.5p}}
	}
	e {
	    ttk::style configure $style.Tab -padding     {1.5p 4.5p 1.5p 4.5p}
	    ttk::style map $style.Tab -padding {selected {1.5p 4.5p 3p   4.5p}}
	}
	default {
	    ttk::style configure $style.Tab -padding     {4.5p 1.5p 4.5p 1.5p}
	    ttk::style map $style.Tab -padding {selected {4.5p 3p   4.5p 1.5p}}
	}
    }
}
