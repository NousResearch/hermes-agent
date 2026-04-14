import json,sys

# Reads JSON array from stdin and prints one alert text per line.
# Falls back to JSON string for unknown shapes.

def main():
    try:
        data = json.load(sys.stdin)
    except Exception as e:
        sys.stderr.write(f"parse_error: {e}\n")
        sys.exit(2)

    if data is None:
        return

    # Allow either a list of alerts or a single alert object
    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        print(json.dumps(data, ensure_ascii=False))
        return

    for a in data:
        text = None
        if isinstance(a, dict):
            for k in ("text", "message", "alert_text", "body", "title"):
                v = a.get(k)
                if isinstance(v, str) and v.strip():
                    text = v.strip()
                    break
            if text is None:
                text = json.dumps(a, ensure_ascii=False)
        else:
            text = str(a)

        if isinstance(text, str) and text.strip():
            print(text.strip())

if __name__ == "__main__":
    main()
