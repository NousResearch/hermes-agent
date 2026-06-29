# Color Themes Renderer

Subagent prompt. Produce one self-contained HTML page at the supplied `.working/color-themes-{n}.html` path showing 4-6 distinct theme variations side by side so the user can pick.

Each variation: header (name + one-line emotional register), token chips for every semantic role decided so far, and one realistic UI snippet using the palette (content drawn from the conversation, not lorem). Include light and dark side-by-side when both modes are in scope. Avoid near-identical pastels — variations must differ in register, not just hue.

Inline CSS only, system font stack, no JS, no network. Document concrete hex values in `<style>` comments per variation so the user can lift them if they pick that theme. The spine itself stays semantic.

Return to the parent: file path, one-line per variation, mode coverage. Do not dump HTML into the parent context. If interactive, open the file with `python3 -c "import webbrowser, pathlib; webbrowser.open(pathlib.Path('PATH').resolve().as_uri())"`.
