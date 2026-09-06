cd "$LOCALAPPDATA/hermes/drp" && grep -n "def load_state" drp_controller.py && sed -n "$(grep -n 'def load_state' drp_controller.py | cut -d: -f1),+14p" drp_controller.py
