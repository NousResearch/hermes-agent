# icons.tcl --
#
#	A set of stock icons for use in Tk dialogs. The icons used here
#	were provided by the Vimix Icon Theme project, which provides a
#	unified set of high quality icons licensed under the
#	Creative Commons Attribution Share-Alike license
#	(https://creativecommons.org/licenses/by-sa/4.0/)
#
#	See https://github.com/vinceliuice/vimix-icon-theme
#
# Copyright © 2009 Pat Thoyts <patthoyts@users.sourceforge.net>
# Copyright © 2022 Harald Oehlmann <harald.oehlmann@elmicron.de>
# Copyright © 2022 Csaba Nemethi <csaba.nemethi@t-online.de>

namespace eval ::tk::icons {}

variable ::tk::svgFmt [list svg -scale [expr {[::tk::ScalingPct] / 100.0}]]

image create photo ::tk::icons::error -format $::tk::svgFmt -data {
    <?xml version="1.0" encoding="UTF-8"?>
    <svg width="32" height="32" version="1.1" xmlns="http://www.w3.org/2000/svg">
     <circle cx="16" cy="16" r="16" fill="#d32f2f"/>
     <g transform="rotate(45,16,16)" fill="#fff">
      <rect x="6" y="14" width="20" height="4"/>
      <rect x="14" y="6" width="4" height="20"/>
     </g>
    </svg>
}

image create photo ::tk::icons::warning -format $::tk::svgFmt -data {
    <?xml version="1.0" encoding="UTF-8"?>
    <svg width="32" height="32" version="1.1" xmlns="http://www.w3.org/2000/svg">
     <circle cx="16" cy="16" r="16" fill="#f67400"/>
     <circle cx="16" cy="24" r="2" fill="#fff"/>
     <path d="m14 20h4v-14h-4z" fill="#fff"/>
    </svg>
}

image create photo ::tk::icons::information -format $::tk::svgFmt -data {
    <?xml version="1.0" encoding="UTF-8"?>
    <svg width="32" height="32" version="1.1" xmlns="http://www.w3.org/2000/svg">
     <circle cx="16" cy="16" r="16" fill="#2091df"/>
     <circle cx="16" cy="8" r="2" fill="#fff"/>
     <path d="m14 12h4v14h-4z" fill="#fff"/>
    </svg>
}

image create photo ::tk::icons::question -format $::tk::svgFmt -data {
    <?xml version="1.0" encoding="UTF-8"?>
    <svg width="32" height="32" version="1.1" xmlns="http://www.w3.org/2000/svg">
     <circle cx="16" cy="16" r="16" fill="#5c6bc0"/>
     <path d="m17.6 27.2h-3.2v-3.2h3.2zm3.312-12.4-1.44 1.472c-1.152 1.168-1.872 2.128-1.872 4.528h-3.2v-0.8c0-1.76 0.72-3.36 1.872-4.528l1.984-2.016a3.128 3.128 0 0 0 0.944-2.256c0-1.76-1.44-3.2-3.2-3.2s-3.2 1.44-3.2 3.2h-3.2c0-3.536 2.864-6.4 6.4-6.4s6.4 2.864 6.4 6.4c0 1.408-0.576 2.688-1.488 3.6z" fill="#fff"/>
    </svg>
}
