"Zoom a window to maximum height."

import re
import sys
import tkinter


class WmInfoGatheringError(Exception):
    pass


class ZoomHeight:
    # Cached values for maximized window dimensions, one for each set
    # of screen dimensions.
    _max_height_and_y_coords = {}

    def __init__(self, editwin):
        self.editwin = editwin
        self.top = self.editwin.top

    def zoom_height_event(self, event=None):
        zoomed = self.zoom_height()

        if zoomed is None:
            self.top.bell()
        else:
            menu_status = 'Restore' if zoomed else 'Zoom'
            self.editwin.update_menu_label(menu='options', index='* Height',
                                           label=f'{menu_status} Height')

        return "break"

    def zoom_height(self):
        top = self.top

        width, height, x, y = get_window_geometry(top)

        if top.wm_state() != 'normal':
            # Can't zoom/restore window height for windows not in the 'normal'
            # state, e.g. maximized and full-screen windows.
            return None

        try:
            maxheight, maxy = self.get_max_height_and_y_coord()
        except WmInfoGatheringError:
            return None

        if height != maxheight:
            # Maximize the window's height.
            set_window_geometry(top, (width, maxheight, x, maxy))
            return True
        else:
            # Restore the window's height.
            #
            # .wm_geometry('') makes the window revert to the size requested
            # by the widgets it contains.
            top.wm_geometry('')
            return False

    def get_max_height_and_y_coord(self):
        top = self.top

        screen_dimensions = (top.winfo_screenwidth(),
                             top.winfo_screenheight())
        if screen_dimensions not in self._max_height_and_y_coords:
            orig_state = top.wm_state()

            # Get window geometry info for maximized windows.
            try:
                top.wm_state('zoomed')
            except tkinter.TclError:
                # The 'zoomed' state is not supported by some esoteric WMs,
                # such as Xvfb.
                raise WmInfoGatheringError(
                    'Failed getting geometry of maximized windows, because ' +
                    'the "zoomed" window state is unavailable.')
            top.update()
            maxwidth, maxheight, maxx, maxy = get_window_geometry(top)
            if sys.platform == 'win32':
                # On Windows, the returned Y coordinate is the one before
                # maximizing, so we use 0 which is correct unless a user puts
                # their dock on the top of the screen (very rare).
                maxy = 0
            maxrooty = top.winfo_rooty()

            # Get the "root y" coordinate for non-maximized windows with their
            # y coordinate set to that of maximized windows.  This is needed
            # to properly handle different title bar heights for non-maximized
            # vs. maximized windows, as seen e.g. in Windows 10.
            top.wm_state('normal')
            top.update()
            orig_geom = get_window_geometry(top)
            max_y_geom = orig_geom[:3] + (maxy,)
            set_window_geometry(top, max_y_geom)
            top.update()
            max_y_geom_rooty = top.winfo_rooty()

            # Adjust the maximum window height to account for the different
            # title bar heights of non-maximized vs. maximized windows.
            maxheight += maxrooty - max_y_geom_rooty

            self._max_height_and_y_coords[screen_dimensions] = maxheight, maxy

            set_window_geometry(top, orig_geom)
            top.wm_state(orig_state)

        return self._max_height_and_y_coords[screen_dimensions]


def get_window_geometry(top):
    geom = top.wm_geometry()
    m = re.match(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)", geom)
    return tuple(map(int, m.groups()))


def set_window_geometry(top, geometry):
    top.wm_geometry("{:d}x{:d}+{:d}+{:d}".format(*geometry))


if __name__ == "__main__":
    from unittest import main
    main('idlelib.idle_test.test_zoomheight', verbosity=2, exit=False)

    # Add htest?
