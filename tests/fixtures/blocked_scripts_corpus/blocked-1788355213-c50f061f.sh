grep -n "base_url" "$LOCALAPPDATA/hermes/config.yaml"; sed -n "$(grep -n 'base_url' "$LOCALAPPDATA/hermes/config.yaml" | tail -1 | cut -d: -f1),+4p" "$LOCALAPPDATA/hermes/config.yaml"
