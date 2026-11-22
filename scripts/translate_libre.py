import json
import os
import sys

# Ensure local Argos / LibreTranslate models are located in hermes cache
os.environ.setdefault("XDG_DATA_HOME", r"C:\Users\mihai\AppData\Local\hermes\cache\data")
os.environ.setdefault("XDG_CONFIG_HOME", r"C:\Users\mihai\AppData\Local\hermes\cache\config")
os.environ.setdefault("XDG_CACHE_HOME", r"C:\Users\mihai\AppData\Local\hermes\cache\cache")

try:
    import argostranslate.translate
except Exception as e:
    sys.stderr.write(f"ArgosTranslate import error: {e}\n")
    sys.exit(1)

def translate_text(text: str, from_code: str = "en", to_code: str = "ro") -> str:
    if not text or not text.strip():
        return text
    try:
        return argostranslate.translate.translate(text, from_code, to_code)
    except Exception as e:
        sys.stderr.write(f"Translation error: {e}\n")
        return text

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        payload = json.load(sys.stdin)
        results = [translate_text(item) for item in payload]
        json.dump(results, sys.stdout, ensure_ascii=False)
    elif len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        print(translate_text(text))
    else:
        text = sys.stdin.read()
        print(translate_text(text), end="")
