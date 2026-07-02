"""Line numbering implementation for IDLE as an extension.
Includes BaseSideBar which can be extended for other sidebar based extensions
"""
import contextlib
import functools
import itertools

import tkinter as tk
from tkinter.font import Font
from idlelib.config import idleConf
from idlelib.delegator import Delegator
from idlelib import macosx


def get_lineno(text, index):
    """Return the line number of an index in a Tk text widget."""
    text_index = text.index(index)
    return int(float(text_index)) if text_index else None


def get_end_linenumber(text):
    """Return the number of the last line in a Tk text widget."""
    return get_lineno(text, 'end-1c')


def get_displaylines(text, index):
    """Display height, in lines, of a logical line in a Tk text widget."""
    res = text.count(f"{index} linestart",
                     f"{index} lineend",
                     "displaylines")
    return res[0] if res else 0

def get_widget_padding(widget):
    """Get the total padding of a Tk widget, including its border."""
    # TODO: use also in codecontext.py
    manager = widget.winfo_manager()
    if manager == 'pack':
        info = widget.pack_info()
    elif manager == 'grid':
        info = widget.grid_info()
    else:
        raise ValueError(f"Unsupported geometry manager: {manager}")

    # All values are passed through getint(), since some
    # values may be pixel objects, which can't simply be added to ints.
    padx = sum(map(widget.tk.getint, [
        info['padx'],
        widget.cget('padx'),
        widget.cget('border'),
    ]))
    pady = sum(map(widget.tk.getint, [
        info['pady'],
        widget.cget('pady'),
        widget.cget('border'),
    ]))
    return padx, pady


@contextlib.contextmanager
def temp_enable_text_widget(text):
    text.configure(state=tk.NORMAL)
    try:
        yield
    finally:
        text.configure(state=tk.DISABLED)


class BaseSideBar:
    """A base class for sidebars using Text."""
    def __init__(self, editwin):
        self.editwin = editwin
        self.parent = editwin.text_frame
        self.text = editwin.text

        self.is_shown = False

        self.main_widget = self.init_widgets()

        self.bind_events()

        self.update_font()
        self.update_colors()

    def init_widgets(self):
        """Initialize the sidebar's widgets, returning the main widget."""
        raise NotImplementedError

    def update_font(self):
        """Update the sidebar text font, usually after config changes."""
        raise NotImplementedError

    def update_colors(self):
        """Update the sidebar text colors, usually after config changes."""
        raise NotImplementedError

    def grid(self):
        """Layout the widget, always using grid layout."""
        raise NotImplementedError

    def show_sidebar(self):
        if not self.is_shown:
            self.grid()
            self.is_shown = True

    def hide_sidebar(self):
        if self.is_shown:
            self.main_widget.grid_forget()
            self.is_shown = False

    def yscroll_event(self, *args, **kwargs):
        """Hook for vertical scrolling for sub-classes to override."""
        raise NotImplementedError

    def redirect_yscroll_event(self, *args, **kwargs):
        """Redirect vertical scrolling to the main editor text widget.

        The scroll bar is also updated.
        """
        self.editwin.vbar.set(*args)
        return self.yscroll_event(*args, **kwargs)

    def redirect_focusin_event(self, event):
        """Redirect focus-in events to the main editor text widget."""
        self.text.focus_set()
        return 'break'

    def redirect_mousebutton_event(self, event, event_name):
        """Redirect mouse button events to the main editor text widget."""
        self.text.focus_set()
        self.text.event_generate(event_name, x=0, y=event.y)
        return 'break'

    def redirect_mousewheel_event(self, event):
        """Redirect mouse wheel events to the editwin text widget."""
        self.text.event_generate('<MouseWheel>',
                                 x=0, y=event.y, delta=event.delta)
        return 'break'

    def bind_events(self):
        self.text['yscrollcommand'] = self.redirect_yscroll_event

        # Ensure focus is always redirected to the main editor text widget.
        self.main_widget.bind('<FocusIn>', self.redirect_focusin_event)

        # Redirect mouse scrolling to the main editor text widget.
        #
        # Note that without this, scrolling with the mouse only scrolls
        # the line numbers.
        self.main_widget.bind('<MouseWheel>', self.redirect_mousewheel_event)

        # Redirect mouse button events to the main editor text widget,
        # except for the left mouse button (1).
        #
        # Note: X-11 sends Button-4 and Button-5 events for the scroll wheel.
        def bind_mouse_event(event_name, target_event_name):
            handler = functools.partial(self.redirect_mousebutton_event,
                                        event_name=target_event_name)
            self.main_widget.bind(event_name, handler)

        for button in [2, 3, 4, 5]:
            for event_name in (f'<Button-{button}>',
                               f'<ButtonRelease-{button}>',
                               f'<B{button}-Motion>',
                               ):
                bind_mouse_event(event_name, target_event_name=event_name)

            # Convert double- and triple-click events to normal click events,
            # since event_generate() doesn't allow generating such events.
            for event_name in (f'<Double-Button-{button}>',
                               f'<Triple-Button-{button}>',
                               ):
                bind_mouse_event(event_name,
                                 target_event_name=f'<Button-{button}>')

        # start_line is set upon <Button-1> to allow selecting a range of rows
        # by dragging.  It is cleared upon <ButtonRelease-1>.
        start_line = None

        # last_y is initially set upon <B1-Leave> and is continuously updated
        # upon <B1-Motion>, until <B1-Enter> or the mouse button is released.
        # It is used in text_auto_scroll(), which is called repeatedly and
        # does have a mouse event available.
        last_y = None

        # auto_scrolling_after_id is set whenever text_auto_scroll is
        # scheduled via .after().  It is used to stop the auto-scrolling
        # upon <B1-Enter>, as well as to avoid scheduling the function several
        # times in parallel.
        auto_scrolling_after_id = None

        def drag_update_selection_and_insert_mark(y_coord):
            """Helper function for drag and selection event handlers."""
            lineno = get_lineno(self.text, f"@0,{y_coord}")
            a, b = sorted([start_line, lineno])
            self.text.tag_remove("sel", "1.0", "end")
            self.text.tag_add("sel", f"{a}.0", f"{b+1}.0")
            self.text.mark_set("insert",
                               f"{lineno if lineno == a else lineno + 1}.0")

        def b1_mousedown_handler(event):
            nonlocal start_line
            nonlocal last_y
            start_line = int(float(self.text.index(f"@0,{event.y}")))
            last_y = event.y

            drag_update_selection_and_insert_mark(event.y)
        self.main_widget.bind('<Button-1>', b1_mousedown_handler)

        def b1_mouseup_handler(event):
            # On mouse up, we're no longer dragging.  Set the shared persistent
            # variables to None to represent this.
            nonlocal start_line
            nonlocal last_y
            start_line = None
            last_y = None
            self.text.event_generate('<ButtonRelease-1>', x=0, y=event.y)
        self.main_widget.bind('<ButtonRelease-1>', b1_mouseup_handler)

        def b1_drag_handler(event):
            nonlocal last_y
            if last_y is None:  # i.e. if not currently dragging
                return
            last_y = event.y
            drag_update_selection_and_insert_mark(event.y)
        self.main_widget.bind('<B1-Motion>', b1_drag_handler)

        def text_auto_scroll():
            """Mimic Text auto-scrolling when dragging outside of it."""
            # See: https://github.com/tcltk/tk/blob/064ff9941b4b80b85916a8afe86a6c21fd388b54/library/text.tcl#L670
            nonlocal auto_scrolling_after_id
            y = last_y
            if y is None:
                self.main_widget.after_cancel(auto_scrolling_after_id)
                auto_scrolling_after_id = None
                return
            elif y < 0:
                self.text.yview_scroll(-1 + y, 'pixels')
                drag_update_selection_and_insert_mark(y)
            elif y > self.main_widget.winfo_height():
                self.text.yview_scroll(1 + y - self.main_widget.winfo_height(),
                                       'pixels')
                drag_update_selection_and_insert_mark(y)
            auto_scrolling_after_id = \
                self.main_widget.after(50, text_auto_scroll)

        def b1_leave_handler(event):
            # Schedule the initial call to text_auto_scroll(), if not already
            # scheduled.
            nonlocal auto_scrolling_after_id
            if auto_scrolling_after_id is None:
                nonlocal last_y
                last_y = event.y
                auto_scrolling_after_id = \
                    self.main_widget.after(0, text_auto_scroll)
        self.main_widget.bind('<B1-Leave>', b1_leave_handler)

        def b1_enter_handler(event):
            # Cancel the scheduling of text_auto_scroll(), if it exists.
            nonlocal auto_scrolling_after_id
            if auto_scrolling_after_id is not None:
                self.main_widget.after_cancel(auto_scrolling_after_id)
                auto_scrolling_after_id = None
        self.main_widget.bind('<B1-Enter>', b1_enter_handler)


class EndLineDelegator(Delegator):
    """Generate callbacks with the current end line number.

    The provided callback is called after every insert and delete.
    """
    def __init__(self, changed_callback):
        Delegator.__init__(self)
        self.changed_callback = changed_callback

    def insert(self, index, chars, tags=None):
        self.delegate.insert(index, chars, tags)
        self.changed_callback(get_end_linenumber(self.delegate))

    def delete(self, index1, index2=None):
        self.delegate.delete(index1, index2)
        self.changed_callback(get_end_linenumber(self.delegate))


class LineNumbers(BaseSideBar):
    """Line numbers support for editor windows."""
    def __init__(self, editwin):
        super().__init__(editwin)

        end_line_delegator = EndLineDelegator(self.update_sidebar_text)
        # Insert the delegator after the undo delegator, so that line numbers
        # are properly updated after undo and redo actions.
        self.editwin.per.insertfilterafter(end_line_delegator,
                                           after=self.editwin.undo)

    def init_widgets(self):
        _padx, pady = get_widget_padding(self.text)
        self.sidebar_text = tk.Text(self.parent, width=1, wrap=tk.NONE,
                                    padx=2, pady=pady,
                                    borderwidth=0, highlightthickness=0)
        self.sidebar_text.config(state=tk.DISABLED)

        self.prev_end = 1
        self._sidebar_width_type = type(self.sidebar_text['width'])
        with temp_enable_text_widget(self.sidebar_text):
            self.sidebar_text.insert('insert', '1', 'linenumber')
        self.sidebar_text.config(takefocus=False, exportselection=False)
        self.sidebar_text.tag_config('linenumber', justify=tk.RIGHT)

        end = get_end_linenumber(self.text)
        self.update_sidebar_text(end)

        return self.sidebar_text

    def grid(self):
        self.sidebar_text.grid(row=1, column=0, sticky=tk.NSEW)

    def update_font(self):
        font = idleConf.GetFont(self.text, 'main', 'EditorWindow')
        self.sidebar_text['font'] = font

    def update_colors(self):
        """Update the sidebar text colors, usually after config changes."""
        colors = idleConf.GetHighlight(idleConf.CurrentTheme(), 'linenumber')
        foreground = colors['foreground']
        background = colors['background']
        self.sidebar_text.config(
            fg=foreground, bg=background,
            selectforeground=foreground, selectbackground=background,
            inactiveselectbackground=background,
        )

    def update_sidebar_text(self, end):
        """
        Perform the following action:
        Each line sidebar_text contains the linenumber for that line
        Synchronize with editwin.text so that both sidebar_text and
        editwin.text contain the same number of lines"""
        if end == self.prev_end:
            return

        width_difference = len(str(end)) - len(str(self.prev_end))
        if width_difference:
            cur_width = int(float(self.sidebar_text['width']))
            new_width = cur_width + width_difference
            self.sidebar_text['width'] = self._sidebar_width_type(new_width)

        with temp_enable_text_widget(self.sidebar_text):
            if end > self.prev_end:
                new_text = '\n'.join(itertools.chain(
                    [''],
                    map(str, range(self.prev_end + 1, end + 1)),
                ))
                self.sidebar_text.insert(f'end -1c', new_text, 'linenumber')
            else:
                self.sidebar_text.delete(f'{end+1}.0 -1c', 'end -1c')

        self.prev_end = end

    def yscroll_event(self, *args, **kwargs):
        self.sidebar_text.yview_moveto(args[0])
        return 'break'


class WrappedLineHeightChangeDelegator(Delegator):
    def __init__(self, callback):
        """
        callback - Callable, will be called when an insert, delete or replace
                   action on the text widget may require updating the shell
                   sidebar.
        """
        Delegator.__init__(self)
        self.callback = callback

    def insert(self, index, chars, tags=None):
        is_single_line = '\n' not in chars
        if is_single_line:
            before_displaylines = get_displaylines(self, index)

        self.delegate.insert(index, chars, tags)

        if is_single_line:
            after_displaylines = get_displaylines(self, index)
            if after_displaylines == before_displaylines:
                return  # no need to update the sidebar

        self.callback()

    def delete(self, index1, index2=None):
        if index2 is None:
            index2 = index1 + "+1c"
        is_single_line = get_lineno(self, index1) == get_lineno(self, index2)
        if is_single_line:
            before_displaylines = get_displaylines(self, index1)

        self.delegate.delete(index1, index2)

        if is_single_line:
            after_displaylines = get_displaylines(self, index1)
            if after_displaylines == before_displaylines:
                return  # no need to update the sidebar

        self.callback()


class ShellSidebar(BaseSideBar):
    """Sidebar for the PyShell window, for prompts etc."""
    def __init__(self, editwin):
        self.canvas = None
        self.line_prompts = {}

        super().__init__(editwin)

        change_delegator = \
            WrappedLineHeightChangeDelegator(self.change_callback)
        # Insert the TextChangeDelegator after the last delegator, so that
        # the sidebar reflects final changes to the text widget contents.
        d = self.editwin.per.top
        if d.delegate is not self.text:
            while d.delegate is not self.editwin.per.bottom:
                d = d.delegate
        self.editwin.per.insertfilterafter(change_delegator, after=d)

        self.is_shown = True

    def init_widgets(self):
        self.canvas = tk.Canvas(self.parent, width=30,
                                borderwidth=0, highlightthickness=0,
                                takefocus=False)
        self.update_sidebar()
        self.grid()
        return self.canvas

    def bind_events(self):
        super().bind_events()

        self.main_widget.bind(
            # AquaTk defines <2> as the right button, not <3>.
            "<Button-2>" if macosx.isAquaTk() else "<Button-3>",
            self.context_menu_event,
        )

    def context_menu_event(self, event):
        rmenu = tk.Menu(self.main_widget, tearoff=0)
        has_selection = bool(self.text.tag_nextrange('sel', '1.0'))
        def mkcmd(eventname):
            return lambda: self.text.event_generate(eventname)
        rmenu.add_command(label='Copy',
                          command=mkcmd('<<copy>>'),
                          state='normal' if has_selection else 'disabled')
        rmenu.add_command(label='Copy with prompts',
                          command=mkcmd('<<copy-with-prompts>>'),
                          state='normal' if has_selection else 'disabled')
        rmenu.tk_popup(event.x_root, event.y_root)
        return "break"

    def grid(self):
        self.canvas.grid(row=1, column=0, sticky=tk.NSEW, padx=2, pady=0)

    def change_callback(self):
        if self.is_shown:
            self.update_sidebar()

    def update_sidebar(self):
        text = self.text
        text_tagnames = text.tag_names
        canvas = self.canvas
        line_prompts = self.line_prompts = {}

        canvas.delete(tk.ALL)

        index = text.index("@0,0")
        if index.split('.', 1)[1] != '0':
            index = text.index(f'{index}+1line linestart')
        while (lineinfo := text.dlineinfo(index)) is not None:
            y = lineinfo[1]
            prev_newline_tagnames = text_tagnames(f"{index} linestart -1c")
            prompt = (
                '>>>' if "console" in prev_newline_tagnames else
                '...' if "stdin" in prev_newline_tagnames else
                None
            )
            if prompt:
                canvas.create_text(2, y, anchor=tk.NW, text=prompt,
                                   font=self.font, fill=self.colors[0])
                lineno = get_lineno(text, index)
                line_prompts[lineno] = prompt
            index = text.index(f'{index}+1line')

    def yscroll_event(self, *args, **kwargs):
        """Redirect vertical scrolling to the main editor text widget.

        The scroll bar is also updated.
        """
        self.change_callback()
        return 'break'

    def update_font(self):
        """Update the sidebar text font, usually after config changes."""
        font = idleConf.GetFont(self.text, 'main', 'EditorWindow')
        tk_font = Font(self.text, font=font)
        char_width = max(tk_font.measure(char) for char in ['>', '.'])
        self.canvas.configure(width=char_width * 3 + 4)
        self.font = font
        self.change_callback()

    def update_colors(self):
        """Update the sidebar text colors, usually after config changes."""
        linenumbers_colors = idleConf.GetHighlight(idleConf.CurrentTheme(), 'linenumber')
        prompt_colors = idleConf.GetHighlight(idleConf.CurrentTheme(), 'console')
        foreground = prompt_colors['foreground']
        background = linenumbers_colors['background']
        self.colors = (foreground, background)
        self.canvas.configure(background=background)
        self.change_callback()


def _sidebar_number_scrolling(parent):  # htest #
    from idlelib.idle_test.test_sidebar import Dummy_editwin

    top = tk.Toplevel(parent)
    text_frame = tk.Frame(top)
    text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    text_frame.rowconfigure(1, weight=1)
    text_frame.columnconfigure(1, weight=1)

    font = idleConf.GetFont(top, 'main', 'EditorWindow')
    text = tk.Text(text_frame, width=80, height=24, wrap=tk.NONE, font=font)
    text.grid(row=1, column=1, sticky=tk.NSEW)

    editwin = Dummy_editwin(text)
    editwin.vbar = tk.Scrollbar(text_frame)

    linenumbers = LineNumbers(editwin)
    linenumbers.show_sidebar()

    text.insert('1.0', '\n'.join('a'*i for i in range(1, 101)))


if __name__ == '__main__':
    from unittest import main
    main('idlelib.idle_test.test_sidebar', verbosity=2, exit=False)

    from idlelib.idle_test.htest import run
    run(_sidebar_number_scrolling)
