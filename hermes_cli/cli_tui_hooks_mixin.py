"""Protected TUI extension hooks for wrapper CLIs (cli.py god-file slice R5).

Hosts the four protected TUI extension hooks lifted out of ``cli.py``'s
``HermesCLI`` class so wrapper CLIs can extend the interactive TUI without
overriding ``run()``:

  - _apply_tui_skin_style
  - _get_extra_tui_widgets
  - _register_extra_tui_keybindings
  - _build_tui_layout_children

``HermesCLI`` inherits ``CLITUIHooksMixin``; every ``self.<hook>`` call
resolves unchanged via the MRO — behavior-neutral. The mixin touches only
``self`` state (getattr-guarded) plus prompt_toolkit, and never imports
``cli`` at module level (no import cycle).
"""

from prompt_toolkit.layout import Window
from prompt_toolkit.styles import Style as PTStyle


class CLITUIHooksMixin:
    """Protected TUI extension hooks wrapper CLIs override without touching run()."""

    def _apply_tui_skin_style(self) -> bool:
        """Refresh prompt_toolkit styling for a running interactive TUI."""
        if not getattr(self, "_app", None) or not getattr(self, "_tui_style_base", None):
            return False
        self._app.style = PTStyle.from_dict(self._build_tui_style_dict())
        self._invalidate(min_interval=0.0)
        return True

    # --- Protected TUI extension hooks for wrapper CLIs ---

    def _get_extra_tui_widgets(self) -> list:
        """Return extra prompt_toolkit widgets to insert into the TUI layout.

        Wrapper CLIs can override this to inject widgets (e.g. a mini-player,
        overlay menu) into the layout without overriding ``run()``.  Widgets
        are inserted between the spacer and the status bar.
        """
        return []

    def _register_extra_tui_keybindings(self, kb, *, input_area) -> None:
        """Register extra keybindings on the TUI ``KeyBindings`` object.

        Wrapper CLIs can override this to add keybindings (e.g. transport
        controls, modal shortcuts) without overriding ``run()``.

        Parameters
        ----------
        kb : KeyBindings
            The active keybinding registry for the prompt_toolkit application.
        input_area : TextArea
            The main input widget, for wrappers that need to inspect or
            manipulate user input from a keybinding handler.
        """

    def _build_tui_layout_children(
        self,
        *,
        sudo_widget,
        secret_widget,
        approval_widget,
        slash_confirm_widget=None,
        clarify_widget,
        model_picker_widget=None,
        spinner_widget=None,
        spacer,
        status_bar,
        input_rule_top,
        image_bar,
        input_area,
        input_rule_bot,
        voice_status_bar,
        completions_menu,
    ) -> list:
        """Assemble the ordered list of children for the root ``HSplit``.

        Wrapper CLIs typically override ``_get_extra_tui_widgets`` instead of
        this method.  Override this only when you need full control over widget
        ordering.
        """
        return [
            item for item in [
                Window(height=0),
                sudo_widget,
                secret_widget,
                approval_widget,
                slash_confirm_widget,
                clarify_widget,
                model_picker_widget,
                spinner_widget,
                spacer,
                *self._get_extra_tui_widgets(),
                getattr(self, "_pet_widget", None),
                getattr(self, "_stash_panel_widget", None),
                status_bar,
                input_rule_top,
                image_bar,
                input_area,
                input_rule_bot,
                voice_status_bar,
                completions_menu,
            ] if item is not None
        ]
