#!/usr/bin/env python3
"""render_draft_html.py

Reusable renderer for the GitHub draft review procedure: draw a self-contained,
GitHub-dark, single-file HTML draft for human review before posting to GitHub.

Self-contained renderer for the GitHub draft review skill.

Runtime deps: stdlib + python `markdown` + `pygments` + the github-markdown-css
stylesheet embedded below (MIT, GitHub / Sindre Sorhus github-markdown-css).
No fetches at view time.

Palette (exact, from the SOP):
  page/gutter bg #0d1117   gutter border  #30363d
  base text      #c9d1d9   headings       #f0f6fc
  muted          #8b949e   accent links   #58a6ff
  code/fence bg  #161b22   fence border   #30363d
  table header   #161b22   table borders  #30363d
  zebra alt row  #161b22   blockquote     #30363d / #8b949e
  badge text     #f85149   badge bg       #3d1d20, border #f85149
  fence lang     #8b949e   code content   #c9d1d9
No other theme.
"""
from __future__ import annotations

import argparse
import difflib
import html as _html
import os
import re
import sys
import webbrowser
from pathlib import Path

try:
    import markdown as _md
    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import get_lexer_by_name, TextLexer
    has_pygments = True
except ImportError:  # pragma: no cover - guarded, not expected
    _md = None
    has_pygments = False

__version__ = "1.2.0"  # --owners/--state keep poster attribution in the chrome
DEFAULT_OWNERS = "draft author / independent proofread / final review"
DEFAULT_STATE = "awaiting human review"

# --------------------------------------------------------------------------
# Embedded github-markdown-css (fetched from GitHub's public stylesheet).
# MIT License — Copyright (c) 2017 Sindre Sorhus / GitHub Primer.
# Snapshot embedded here so the renderer needs no network access.
# --------------------------------------------------------------------------
GITHUB_MARKDOWN_CSS = """.markdown-body {
  --base-size-16: 1rem;
  --base-size-24: 1.5rem;
  --base-size-4: 0.25rem;
  --base-size-40: 2.5rem;
  --base-size-8: 0.5rem;
  --base-text-weight-medium: 500;
  --base-text-weight-normal: 400;
  --base-text-weight-semibold: 600;
  --fontStack-monospace: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace;
  --fontStack-sansSerif: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
  --fgColor-accent: Highlight;
}
@media (prefers-color-scheme: dark) {
  .markdown-body, [data-theme="dark"] {
    /*dark */
    color-scheme: dark;
    --fgColor-accent: #4493f8;
    --bgColor-attention-muted: #bb800926;
    --bgColor-default: #0d1117;
    --bgColor-muted: #151b23;
    --bgColor-neutral-muted: #656c7633;
    --borderColor-accent-emphasis: #1f6feb;
    --borderColor-attention-emphasis: #9e6a03;
    --borderColor-danger-emphasis: #da3633;
    --borderColor-default: #3d444d;
    --borderColor-done-emphasis: #8957e5;
    --borderColor-success-emphasis: #238636;
    --color-prettylights-syntax-brackethighlighter-angle: #9198a1;
    --color-prettylights-syntax-brackethighlighter-unmatched: #f85149;
    --color-prettylights-syntax-carriage-return-bg: #b62324;
    --color-prettylights-syntax-carriage-return-text: #f0f6fc;
    --color-prettylights-syntax-comment: #9198a1;
    --color-prettylights-syntax-constant: #79c0ff;
    --color-prettylights-syntax-constant-other-reference-link: #a5d6ff;
    --color-prettylights-syntax-entity: #d2a8ff;
    --color-prettylights-syntax-entity-tag: #7ee787;
    --color-prettylights-syntax-keyword: #ff7b72;
    --color-prettylights-syntax-markup-bold: #f0f6fc;
    --color-prettylights-syntax-markup-changed-bg: #5a1e02;
    --color-prettylights-syntax-markup-changed-text: #ffdfb6;
    --color-prettylights-syntax-markup-deleted-bg: #67060c;
    --color-prettylights-syntax-markup-deleted-text: #ffdcd7;
    --color-prettylights-syntax-markup-heading: #1f6feb;
    --color-prettylights-syntax-markup-ignored-bg: #1158c7;
    --color-prettylights-syntax-markup-ignored-text: #f0f6fc;
    --color-prettylights-syntax-markup-inserted-bg: #033a16;
    --color-prettylights-syntax-markup-inserted-text: #aff5b4;
    --color-prettylights-syntax-markup-italic: #f0f6fc;
    --color-prettylights-syntax-markup-list: #f2cc60;
    --color-prettylights-syntax-meta-diff-range: #d2a8ff;
    --color-prettylights-syntax-storage-modifier-import: #f0f6fc;
    --color-prettylights-syntax-string: #a5d6ff;
    --color-prettylights-syntax-string-regexp: #7ee787;
    --color-prettylights-syntax-sublimelinter-gutter-mark: #3d444d;
    --color-prettylights-syntax-variable: #ffa657;
    --fgColor-attention: #d29922;
    --fgColor-danger: #f85149;
    --fgColor-default: #f0f6fc;
    --fgColor-done: #ab7df8;
    --fgColor-muted: #9198a1;
    --fgColor-success: #3fb950;
    --borderColor-muted: #3d444db3;
    --color-prettylights-syntax-invalid-illegal-bg: var(--bgColor-danger-muted);
    --color-prettylights-syntax-invalid-illegal-text: var(--fgColor-danger);
    --focus-outlineColor: var(--borderColor-accent-emphasis);
    --borderColor-neutral-muted: var(--borderColor-muted);
  }
}
@media (prefers-color-scheme: light) {
  .markdown-body, [data-theme="light"] {
    /*light */
    color-scheme: light;
    --fgColor-danger: #d1242f;
    --bgColor-attention-muted: #fff8c5;
    --bgColor-muted: #f6f8fa;
    --bgColor-neutral-muted: #818b981f;
    --borderColor-accent-emphasis: #0969da;
    --borderColor-attention-emphasis: #9a6700;
    --borderColor-danger-emphasis: #cf222e;
    --borderColor-default: #d1d9e0;
    --borderColor-done-emphasis: #8250df;
    --borderColor-success-emphasis: #1a7f37;
    --color-prettylights-syntax-brackethighlighter-angle: #59636e;
    --color-prettylights-syntax-brackethighlighter-unmatched: #82071e;
    --color-prettylights-syntax-carriage-return-bg: #cf222e;
    --color-prettylights-syntax-carriage-return-text: #f6f8fa;
    --color-prettylights-syntax-comment: #59636e;
    --color-prettylights-syntax-constant: #0550ae;
    --color-prettylights-syntax-constant-other-reference-link: #0a3069;
    --color-prettylights-syntax-entity: #6639ba;
    --color-prettylights-syntax-entity-tag: #0550ae;
    --color-prettylights-syntax-invalid-illegal-text: var(--fgColor-danger);
    --color-prettylights-syntax-keyword: #cf222e;
    --color-prettylights-syntax-markup-changed-bg: #ffd8b5;
    --color-prettylights-syntax-markup-changed-text: #953800;
    --color-prettylights-syntax-markup-deleted-bg: #ffebe9;
    --color-prettylights-syntax-markup-deleted-text: #82071e;
    --color-prettylights-syntax-markup-heading: #0550ae;
    --color-prettylights-syntax-markup-ignored-bg: #0550ae;
    --color-prettylights-syntax-markup-ignored-text: #d1d9e0;
    --color-prettylights-syntax-markup-inserted-bg: #dafbe1;
    --color-prettylights-syntax-markup-inserted-text: #116329;
    --color-prettylights-syntax-markup-list: #3b2300;
    --color-prettylights-syntax-meta-diff-range: #8250df;
    --color-prettylights-syntax-string: #0a3069;
    --color-prettylights-syntax-string-regexp: #116329;
    --color-prettylights-syntax-sublimelinter-gutter-mark: #818b98;
    --color-prettylights-syntax-variable: #953800;
    --fgColor-accent: #0969da;
    --fgColor-attention: #9a6700;
    --fgColor-done: #8250df;
    --fgColor-muted: #59636e;
    --fgColor-success: #1a7f37;
    --bgColor-default: #ffffff;
    --borderColor-muted: #d1d9e0b3;
    --color-prettylights-syntax-invalid-illegal-bg: var(--bgColor-danger-muted);
    --color-prettylights-syntax-markup-bold: #1f2328;
    --color-prettylights-syntax-markup-italic: #1f2328;
    --color-prettylights-syntax-storage-modifier-import: #1f2328;
    --fgColor-default: #1f2328;
    --focus-outlineColor: var(--borderColor-accent-emphasis);
    --borderColor-neutral-muted: var(--borderColor-muted);
  }
}

.markdown-body {
  /** CSS default easing. Use for hover state changes and micro-interactions. */
  /** Accelerating motion. Use for elements exiting the viewport (moving off-screen). */
  /** Smooth acceleration and deceleration. Use for elements moving or morphing within the viewport. */
  /** Decelerating motion. Use for elements entering the viewport or appearing on screen. */
  /** Constant motion with no acceleration. Use for continuous animations like progress bars or loaders. */
  -ms-text-size-adjust: 100%;
  -webkit-text-size-adjust: 100%;
  margin: 0;
  font-weight: var(--base-text-weight-normal, 400);
  color: var(--fgColor-default);
  background-color: var(--bgColor-default);
  font-family: var(--fontStack-sansSerif, -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji");
  font-size: 16px;
  line-height: 1.5;
  word-wrap: break-word;
}

.markdown-body a {
  text-decoration: underline;
  text-underline-offset: .2rem;
}

.markdown-body .octicon {
  display: inline-block;
  fill: currentColor;
  vertical-align: text-bottom;
}

.markdown-body h1:hover .anchor .octicon-link:before,
.markdown-body h2:hover .anchor .octicon-link:before,
.markdown-body h3:hover .anchor .octicon-link:before,
.markdown-body h4:hover .anchor .octicon-link:before,
.markdown-body h5:hover .anchor .octicon-link:before,
.markdown-body h6:hover .anchor .octicon-link:before {
  width: 16px;
  height: 16px;
  content: ' ';
  display: inline-block;
  background-color: currentColor;
  -webkit-mask-image: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' version='1.1' aria-hidden='true'><path fill-rule='evenodd' d='M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z'></path></svg>");
  mask-image: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' version='1.1' aria-hidden='true'><path fill-rule='evenodd' d='M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z'></path></svg>");
}

.markdown-body details,
.markdown-body figcaption,
.markdown-body figure {
  display: block;
}

.markdown-body summary {
  display: list-item;
}

.markdown-body [hidden] {
  display: none !important;
}

.markdown-body a {
  background-color: rgba(0,0,0,0);
  color: var(--fgColor-accent);
  text-decoration: none;
}

.markdown-body abbr[title] {
  border-bottom: none;
  -webkit-text-decoration: underline dotted;
  text-decoration: underline dotted;
}

.markdown-body b,
.markdown-body strong {
  font-weight: var(--base-text-weight-semibold, 600);
}

.markdown-body dfn {
  font-style: italic;
}

.markdown-body h1 {
  margin: .67em 0;
  font-weight: var(--base-text-weight-semibold, 600);
  padding-bottom: .3em;
  font-size: 2em;
  border-bottom: 1px solid var(--borderColor-muted);
}

.markdown-body mark {
  background-color: var(--bgColor-attention-muted);
  color: var(--fgColor-default);
}

.markdown-body small {
  font-size: 90%;
}

.markdown-body sub,
.markdown-body sup {
  font-size: 75%;
  line-height: 0;
  position: relative;
  vertical-align: baseline;
}

.markdown-body sub {
  bottom: -0.25em;
}

.markdown-body sup {
  top: -0.5em;
}

.markdown-body img {
  border-style: none;
  max-width: 100%;
  box-sizing: content-box;
}

.markdown-body code,
.markdown-body kbd,
.markdown-body pre,
.markdown-body samp {
  font-family: monospace;
  font-size: 1em;
}

.markdown-body figure {
  margin: 1em var(--base-size-40);
}

.markdown-body hr {
  box-sizing: content-box;
  overflow: hidden;
  background: rgba(0,0,0,0);
  border-bottom: 1px solid var(--borderColor-muted);
  height: .25em;
  padding: 0;
  margin: var(--base-size-24) 0;
  background-color: var(--borderColor-default);
  border: 0;
}

.markdown-body input {
  font: inherit;
  margin: 0;
  overflow: visible;
  font-family: inherit;
  font-size: inherit;
  line-height: inherit;
}

.markdown-body [type=button],
.markdown-body [type=reset],
.markdown-body [type=submit] {
  -webkit-appearance: button;
  appearance: button;
}

.markdown-body [type=checkbox],
.markdown-body [type=radio] {
  box-sizing: border-box;
  padding: 0;
}

.markdown-body [type=number]::-webkit-inner-spin-button,
.markdown-body [type=number]::-webkit-outer-spin-button {
  height: auto;
}

.markdown-body [type=search]::-webkit-search-cancel-button,
.markdown-body [type=search]::-webkit-search-decoration {
  -webkit-appearance: none;
  appearance: none;
}

.markdown-body ::-webkit-input-placeholder {
  color: inherit;
  opacity: .54;
}

.markdown-body ::-webkit-file-upload-button {
  -webkit-appearance: button;
  appearance: button;
  font: inherit;
}

.markdown-body a:hover {
  text-decoration: underline;
}

.markdown-body ::placeholder {
  color: var(--fgColor-muted);
  opacity: 1;
}

.markdown-body hr::before {
  display: table;
  content: "";
}

.markdown-body hr::after {
  display: table;
  clear: both;
  content: "";
}

.markdown-body table {
  border-spacing: 0;
  border-collapse: collapse;
  display: block;
  width: max-content;
  max-width: 100%;
  overflow: auto;
  font-variant: tabular-nums;
}

.markdown-body td,
.markdown-body th {
  padding: 0;
}

.markdown-body details summary {
  cursor: pointer;
}

.markdown-body a:focus,
.markdown-body [role=button]:focus,
.markdown-body input[type=radio]:focus,
.markdown-body input[type=checkbox]:focus {
  outline: 2px solid var(--focus-outlineColor);
  outline-offset: -2px;
  box-shadow: none;
}

.markdown-body a:focus:not(:focus-visible),
.markdown-body [role=button]:focus:not(:focus-visible),
.markdown-body input[type=radio]:focus:not(:focus-visible),
.markdown-body input[type=checkbox]:focus:not(:focus-visible) {
  outline: solid 1px rgba(0,0,0,0);
}

.markdown-body a:focus-visible,
.markdown-body [role=button]:focus-visible,
.markdown-body input[type=radio]:focus-visible,
.markdown-body input[type=checkbox]:focus-visible {
  outline: 2px solid var(--focus-outlineColor);
  outline-offset: -2px;
  box-shadow: none;
}

.markdown-body a:not([class]):focus,
.markdown-body a:not([class]):focus-visible,
.markdown-body input[type=radio]:focus,
.markdown-body input[type=radio]:focus-visible,
.markdown-body input[type=checkbox]:focus,
.markdown-body input[type=checkbox]:focus-visible {
  outline-offset: 0;
}

.markdown-body kbd {
  display: inline-block;
  padding: var(--base-size-4);
  font: 11px var(--fontStack-monospace, ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace);
  line-height: 10px;
  color: var(--fgColor-default);
  vertical-align: middle;
  background-color: var(--bgColor-muted);
  border: solid 1px var(--borderColor-neutral-muted);
  border-bottom-color: var(--borderColor-neutral-muted);
  border-radius: 6px;
  box-shadow: inset 0 -1px 0 var(--borderColor-neutral-muted);
}

.markdown-body h1,
.markdown-body h2,
.markdown-body h3,
.markdown-body h4,
.markdown-body h5,
.markdown-body h6 {
  margin-top: var(--base-size-24);
  margin-bottom: var(--base-size-16);
  font-weight: var(--base-text-weight-semibold, 600);
  line-height: 1.25;
}

.markdown-body h2 {
  font-weight: var(--base-text-weight-semibold, 600);
  padding-bottom: .3em;
  font-size: 1.5em;
  border-bottom: 1px solid var(--borderColor-muted);
}

.markdown-body h3 {
  font-weight: var(--base-text-weight-semibold, 600);
  font-size: 1.25em;
}

.markdown-body h4 {
  font-weight: var(--base-text-weight-semibold, 600);
  font-size: 1em;
}

.markdown-body h5 {
  font-weight: var(--base-text-weight-semibold, 600);
  font-size: .875em;
}

.markdown-body h6 {
  font-weight: var(--base-text-weight-semibold, 600);
  font-size: .85em;
  color: var(--fgColor-muted);
}

.markdown-body p {
  margin-top: 0;
  margin-bottom: 10px;
}

.markdown-body blockquote {
  margin: 0;
  padding: 0 1em;
  color: var(--fgColor-muted);
  border-left: .25em solid var(--borderColor-default);
}

.markdown-body ul,
.markdown-body ol {
  margin-top: 0;
  margin-bottom: 0;
  padding-left: 2em;
}

.markdown-body ol ol,
.markdown-body ul ol {
  list-style-type: lower-roman;
}

.markdown-body ul ul ol,
.markdown-body ul ol ol,
.markdown-body ol ul ol,
.markdown-body ol ol ol {
  list-style-type: lower-alpha;
}

.markdown-body dd {
  margin-left: 0;
}

.markdown-body tt,
.markdown-body code,
.markdown-body samp {
  font-family: var(--fontStack-monospace, ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace);
  font-size: 12px;
}

.markdown-body pre {
  margin-top: 0;
  margin-bottom: 0;
  font-family: var(--fontStack-monospace, ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace);
  font-size: 12px;
  word-wrap: normal;
}

.markdown-body .octicon {
  display: inline-block;
  overflow: visible !important;
  vertical-align: text-bottom;
  fill: currentColor;
}

.markdown-body input::-webkit-outer-spin-button,
.markdown-body input::-webkit-inner-spin-button {
  margin: 0;
  appearance: none;
}

.markdown-body .mr-2 {
  margin-right: var(--base-size-8, 8px) !important;
}

.markdown-body::before {
  display: table;
  content: "";
}

.markdown-body::after {
  display: table;
  clear: both;
  content: "";
}

.markdown-body>*:first-child {
  margin-top: 0 !important;
}

.markdown-body>*:last-child {
  margin-bottom: 0 !important;
}

.markdown-body a:not([href]) {
  color: inherit;
  text-decoration: none;
}

.markdown-body .absent {
  color: var(--fgColor-danger);
}

.markdown-body .anchor {
  float: left;
  padding-right: var(--base-size-4);
  margin-left: -20px;
  line-height: 1;
}

.markdown-body .anchor:focus {
  outline: none;
}

.markdown-body p,
.markdown-body blockquote,
.markdown-body ul,
.markdown-body ol,
.markdown-body dl,
.markdown-body table,
.markdown-body pre,
.markdown-body details {
  margin-top: 0;
  margin-bottom: var(--base-size-16);
}

.markdown-body blockquote>:first-child {
  margin-top: 0;
}

.markdown-body blockquote>:last-child {
  margin-bottom: 0;
}

.markdown-body h1 .octicon-link,
.markdown-body h2 .octicon-link,
.markdown-body h3 .octicon-link,
.markdown-body h4 .octicon-link,
.markdown-body h5 .octicon-link,
.markdown-body h6 .octicon-link {
  color: var(--fgColor-default);
  vertical-align: middle;
  visibility: hidden;
}

.markdown-body h1:hover .anchor,
.markdown-body h2:hover .anchor,
.markdown-body h3:hover .anchor,
.markdown-body h4:hover .anchor,
.markdown-body h5:hover .anchor,
.markdown-body h6:hover .anchor {
  text-decoration: none;
}

.markdown-body h1:hover .anchor .octicon-link,
.markdown-body h2:hover .anchor .octicon-link,
.markdown-body h3:hover .anchor .octicon-link,
.markdown-body h4:hover .anchor .octicon-link,
.markdown-body h5:hover .anchor .octicon-link,
.markdown-body h6:hover .anchor .octicon-link {
  visibility: visible;
}

.markdown-body h1 tt,
.markdown-body h1 code,
.markdown-body h2 tt,
.markdown-body h2 code,
.markdown-body h3 tt,
.markdown-body h3 code,
.markdown-body h4 tt,
.markdown-body h4 code,
.markdown-body h5 tt,
.markdown-body h5 code,
.markdown-body h6 tt,
.markdown-body h6 code {
  padding: 0 .2em;
  font-size: inherit;
}

.markdown-body summary h1,
.markdown-body summary h2,
.markdown-body summary h3,
.markdown-body summary h4,
.markdown-body summary h5,
.markdown-body summary h6 {
  display: inline-block;
}

.markdown-body summary h1 .anchor,
.markdown-body summary h2 .anchor,
.markdown-body summary h3 .anchor,
.markdown-body summary h4 .anchor,
.markdown-body summary h5 .anchor,
.markdown-body summary h6 .anchor {
  margin-left: -40px;
}

.markdown-body summary h1,
.markdown-body summary h2 {
  padding-bottom: 0;
  border-bottom: 0;
}

.markdown-body ul.no-list,
.markdown-body ol.no-list {
  padding: 0;
  list-style-type: none;
}

.markdown-body ol[type="a s"] {
  list-style-type: lower-alpha;
}

.markdown-body ol[type="A s"] {
  list-style-type: upper-alpha;
}

.markdown-body ol[type="i s"] {
  list-style-type: lower-roman;
}

.markdown-body ol[type="I s"] {
  list-style-type: upper-roman;
}

.markdown-body ol[type="1"] {
  list-style-type: decimal;
}

.markdown-body div>ol:not([type]) {
  list-style-type: decimal;
}

.markdown-body ul ul,
.markdown-body ul ol,
.markdown-body ol ol,
.markdown-body ol ul {
  margin-top: 0;
  margin-bottom: 0;
}

.markdown-body li>p {
  margin-top: var(--base-size-16);
}

.markdown-body li+li {
  margin-top: .25em;
}

.markdown-body dl {
  padding: 0;
}

.markdown-body dl dt {
  padding: 0;
  margin-top: var(--base-size-16);
  font-size: 1em;
  font-style: italic;
  font-weight: var(--base-text-weight-semibold, 600);
}

.markdown-body dl dd {
  padding: 0 var(--base-size-16);
  margin-bottom: var(--base-size-16);
}

.markdown-body table th {
  font-weight: var(--base-text-weight-semibold, 600);
}

.markdown-body table th,
.markdown-body table td {
  padding: 6px 13px;
  border: 1px solid var(--borderColor-default);
}

.markdown-body table td>:last-child {
  margin-bottom: 0;
}

.markdown-body table tr {
  background-color: var(--bgColor-default);
  border-top: 1px solid var(--borderColor-muted);
}

.markdown-body table tr:nth-child(2n) {
  background-color: var(--bgColor-muted);
}

.markdown-body table img {
  background-color: rgba(0,0,0,0);
}

.markdown-body img[align=right] {
  padding-left: 20px;
}

.markdown-body img[align=left] {
  padding-right: 20px;
}

.markdown-body .emoji {
  max-width: none;
  vertical-align: text-top;
  background-color: rgba(0,0,0,0);
}

.markdown-body span.frame {
  display: block;
  overflow: hidden;
}

.markdown-body span.frame>span {
  display: block;
  float: left;
  width: auto;
  padding: 7px;
  margin: 13px 0 0;
  overflow: hidden;
  border: 1px solid var(--borderColor-default);
}

.markdown-body span.frame span img {
  display: block;
  float: left;
}

.markdown-body span.frame span span {
  display: block;
  padding: 5px 0 0;
  clear: both;
  color: var(--fgColor-default);
}

.markdown-body span.align-center {
  display: block;
  overflow: hidden;
  clear: both;
}

.markdown-body span.align-center>span {
  display: block;
  margin: 13px auto 0;
  overflow: hidden;
  text-align: center;
}

.markdown-body span.align-center span img {
  margin: 0 auto;
  text-align: center;
}

.markdown-body span.align-right {
  display: block;
  overflow: hidden;
  clear: both;
}

.markdown-body span.align-right>span {
  display: block;
  margin: 13px 0 0;
  overflow: hidden;
  text-align: right;
}

.markdown-body span.align-right span img {
  margin: 0;
  text-align: right;
}

.markdown-body span.float-left {
  display: block;
  float: left;
  margin-right: 13px;
  overflow: hidden;
}

.markdown-body span.float-left span {
  margin: 13px 0 0;
}

.markdown-body span.float-right {
  display: block;
  float: right;
  margin-left: 13px;
  overflow: hidden;
}

.markdown-body span.float-right>span {
  display: block;
  margin: 13px auto 0;
  overflow: hidden;
  text-align: right;
}

.markdown-body code,
.markdown-body tt {
  padding: .2em .4em;
  margin: 0;
  font-size: 85%;
  white-space: break-spaces;
  background-color: var(--bgColor-neutral-muted);
  border-radius: 6px;
}

.markdown-body code br,
.markdown-body tt br {
  display: none;
}

.markdown-body del code {
  text-decoration: inherit;
}

.markdown-body samp {
  font-size: 85%;
}

.markdown-body pre code {
  font-size: 100%;
}

.markdown-body pre>code {
  padding: 0;
  margin: 0;
  word-break: normal;
  white-space: pre;
  background: rgba(0,0,0,0);
  border: 0;
}

.markdown-body .highlight {
  margin-bottom: var(--base-size-16);
}

.markdown-body .highlight pre {
  margin-bottom: 0;
  word-break: normal;
}

.markdown-body .highlight pre,
.markdown-body pre {
  padding: var(--base-size-16);
  overflow: auto;
  font-size: 85%;
  line-height: 1.45;
  color: var(--fgColor-default);
  background-color: var(--bgColor-muted);
  border-radius: 6px;
}

.markdown-body pre code,
.markdown-body pre tt {
  display: inline;
  padding: 0;
  margin: 0;
  overflow: visible;
  line-height: inherit;
  word-wrap: normal;
  background-color: rgba(0,0,0,0);
  border: 0;
}

.markdown-body .csv-data td,
.markdown-body .csv-data th {
  padding: 5px;
  overflow: hidden;
  font-size: 12px;
  line-height: 1;
  text-align: left;
  white-space: nowrap;
}

.markdown-body .csv-data .blob-num {
  padding: 10px var(--base-size-8) 9px;
  text-align: right;
  background: var(--bgColor-default);
  border: 0;
}

.markdown-body .csv-data tr {
  border-top: 0;
}

.markdown-body .csv-data th {
  font-weight: var(--base-text-weight-semibold, 600);
  background: var(--bgColor-muted);
  border-top: 0;
}

.markdown-body [data-footnote-ref]::before {
  content: "[";
}

.markdown-body [data-footnote-ref]::after {
  content: "]";
}

.markdown-body .footnotes {
  font-size: 12px;
  color: var(--fgColor-muted);
  border-top: 1px solid var(--borderColor-default);
}

.markdown-body .footnotes ol {
  padding-left: var(--base-size-16);
}

.markdown-body .footnotes ol ul {
  display: inline-block;
  padding-left: var(--base-size-16);
  margin-top: var(--base-size-16);
}

.markdown-body .footnotes li {
  position: relative;
}

.markdown-body .footnotes li:target::before {
  position: absolute;
  top: calc(var(--base-size-8)*-1);
  right: calc(var(--base-size-8)*-1);
  bottom: calc(var(--base-size-8)*-1);
  left: calc(var(--base-size-24)*-1);
  pointer-events: none;
  content: "";
  border: 2px solid var(--borderColor-accent-emphasis);
  border-radius: 6px;
}

.markdown-body .footnotes li:target {
  color: var(--fgColor-default);
}

.markdown-body .footnotes .data-footnote-backref g-emoji {
  font-family: monospace;
}

.markdown-body .pl-c {
  color: var(--color-prettylights-syntax-comment);
}

.markdown-body .pl-c1,
.markdown-body .pl-s .pl-v {
  color: var(--color-prettylights-syntax-constant);
}

.markdown-body .pl-e,
.markdown-body .pl-en {
  color: var(--color-prettylights-syntax-entity);
}

.markdown-body .pl-smi,
.markdown-body .pl-s .pl-s1 {
  color: var(--color-prettylights-syntax-storage-modifier-import);
}

.markdown-body .pl-ent {
  color: var(--color-prettylights-syntax-entity-tag);
}

.markdown-body .pl-k {
  color: var(--color-prettylights-syntax-keyword);
}

.markdown-body .pl-s,
.markdown-body .pl-pds,
.markdown-body .pl-s .pl-pse .pl-s1,
.markdown-body .pl-sr,
.markdown-body .pl-sr .pl-cce,
.markdown-body .pl-sr .pl-sre,
.markdown-body .pl-sr .pl-sra {
  color: var(--color-prettylights-syntax-string);
}

.markdown-body .pl-v,
.markdown-body .pl-smw {
  color: var(--color-prettylights-syntax-variable);
}

.markdown-body .pl-bu {
  color: var(--color-prettylights-syntax-brackethighlighter-unmatched);
}

.markdown-body .pl-ii {
  color: var(--color-prettylights-syntax-invalid-illegal-text);
  background-color: var(--color-prettylights-syntax-invalid-illegal-bg);
}

.markdown-body .pl-c2 {
  color: var(--color-prettylights-syntax-carriage-return-text);
  background-color: var(--color-prettylights-syntax-carriage-return-bg);
}

.markdown-body .pl-sr .pl-cce {
  font-weight: bold;
  color: var(--color-prettylights-syntax-string-regexp);
}

.markdown-body .pl-ml {
  color: var(--color-prettylights-syntax-markup-list);
}

.markdown-body .pl-mh,
.markdown-body .pl-mh .pl-en,
.markdown-body .pl-ms {
  font-weight: bold;
  color: var(--color-prettylights-syntax-markup-heading);
}

.markdown-body .pl-mi {
  font-style: italic;
  color: var(--color-prettylights-syntax-markup-italic);
}

.markdown-body .pl-mb {
  font-weight: bold;
  color: var(--color-prettylights-syntax-markup-bold);
}

.markdown-body .pl-md {
  color: var(--color-prettylights-syntax-markup-deleted-text);
  background-color: var(--color-prettylights-syntax-markup-deleted-bg);
}

.markdown-body .pl-mi1 {
  color: var(--color-prettylights-syntax-markup-inserted-text);
  background-color: var(--color-prettylights-syntax-markup-inserted-bg);
}

.markdown-body .pl-mc {
  color: var(--color-prettylights-syntax-markup-changed-text);
  background-color: var(--color-prettylights-syntax-markup-changed-bg);
}

.markdown-body .pl-mi2 {
  color: var(--color-prettylights-syntax-markup-ignored-text);
  background-color: var(--color-prettylights-syntax-markup-ignored-bg);
}

.markdown-body .pl-mdr {
  font-weight: bold;
  color: var(--color-prettylights-syntax-meta-diff-range);
}

.markdown-body .pl-ba {
  color: var(--color-prettylights-syntax-brackethighlighter-angle);
}

.markdown-body .pl-sg {
  color: var(--color-prettylights-syntax-sublimelinter-gutter-mark);
}

.markdown-body .pl-corl {
  text-decoration: underline;
  color: var(--color-prettylights-syntax-constant-other-reference-link);
}

.markdown-body [role=button]:focus:not(:focus-visible),
.markdown-body [role=tabpanel][tabindex="0"]:focus:not(:focus-visible),
.markdown-body button:focus:not(:focus-visible),
.markdown-body summary:focus:not(:focus-visible),
.markdown-body a:focus:not(:focus-visible) {
  outline: none;
  box-shadow: none;
}

.markdown-body [tabindex="0"]:focus:not(:focus-visible),
.markdown-body details-dialog:focus:not(:focus-visible) {
  outline: none;
}

.markdown-body g-emoji {
  display: inline-block;
  min-width: 1ch;
  font-family: "Apple Color Emoji","Segoe UI Emoji","Segoe UI Symbol";
  font-size: 1em;
  font-style: normal !important;
  font-weight: var(--base-text-weight-normal, 400);
  line-height: 1;
  vertical-align: -0.075em;
}

.markdown-body g-emoji img {
  width: 1em;
  height: 1em;
}

.markdown-body a:has(>p,>div,>pre,>blockquote) {
  display: block;
}

.markdown-body a:has(>p,>div,>pre,>blockquote):not(:has(.snippet-clipboard-content,>pre)) {
  width: fit-content;
}

.markdown-body a:has(>p,>div,>pre,>blockquote):has(.snippet-clipboard-content,>pre):focus-visible {
  outline: 2px solid var(--focus-outlineColor);
  outline-offset: 2px;
}

.markdown-body .task-list-item {
  list-style-type: none;
}

.markdown-body .task-list-item label {
  font-weight: var(--base-text-weight-normal, 400);
}

.markdown-body .task-list-item.enabled label {
  cursor: pointer;
}

.markdown-body .task-list-item+.task-list-item {
  margin-top: var(--base-size-4);
}

.markdown-body .task-list-item .handle {
  display: none;
}

.markdown-body .task-list-item-checkbox {
  margin: 0 .2em .25em -1.4em;
  vertical-align: middle;
}

.markdown-body ul:dir(rtl) .task-list-item-checkbox {
  margin: 0 -1.6em .25em .2em;
}

.markdown-body ol:dir(rtl) .task-list-item-checkbox {
  margin: 0 -1.6em .25em .2em;
}

.markdown-body .contains-task-list:hover .task-list-item-convert-container,
.markdown-body .contains-task-list:focus-within .task-list-item-convert-container {
  display: block;
  width: auto;
  height: 24px;
  overflow: visible;
  clip-path: none;
}

.markdown-body ::-webkit-calendar-picker-indicator {
  filter: invert(50%);
}

.markdown-body .markdown-alert {
  padding: var(--base-size-8) var(--base-size-16);
  margin-bottom: var(--base-size-16);
  color: inherit;
  border-left: .25em solid var(--borderColor-default);
}

.markdown-body .markdown-alert>:first-child {
  margin-top: 0;
}

.markdown-body .markdown-alert>:last-child {
  margin-bottom: 0;
}

.markdown-body .markdown-alert .markdown-alert-title {
  display: flex;
  font-weight: var(--base-text-weight-medium, 500);
  align-items: center;
  line-height: 1;
}

.markdown-body .markdown-alert.markdown-alert-note {
  border-left-color: var(--borderColor-accent-emphasis);
}

.markdown-body .markdown-alert.markdown-alert-note .markdown-alert-title {
  color: var(--fgColor-accent);
}

.markdown-body .markdown-alert.markdown-alert-important {
  border-left-color: var(--borderColor-done-emphasis);
}

.markdown-body .markdown-alert.markdown-alert-important .markdown-alert-title {
  color: var(--fgColor-done);
}

.markdown-body .markdown-alert.markdown-alert-warning {
  border-left-color: var(--borderColor-attention-emphasis);
}

.markdown-body .markdown-alert.markdown-alert-warning .markdown-alert-title {
  color: var(--fgColor-attention);
}

.markdown-body .markdown-alert.markdown-alert-tip {
  border-left-color: var(--borderColor-success-emphasis);
}

.markdown-body .markdown-alert.markdown-alert-tip .markdown-alert-title {
  color: var(--fgColor-success);
}

.markdown-body .markdown-alert.markdown-alert-caution {
  border-left-color: var(--borderColor-danger-emphasis);
}

.markdown-body .markdown-alert.markdown-alert-caution .markdown-alert-title {
  color: var(--fgColor-danger);
}

.markdown-body>*:first-child>.heading-element:first-child {
  margin-top: 0 !important;
}

.markdown-body .highlight pre:has(+.zeroclipboard-container) {
  min-height: 52px;
}

"""

# --------------------------------------------------------------------------
# Page chrome + forced GitHub-dark palette on top of the base stylesheet.
# The base stylesheet reads OS prefers-color-scheme; we force dark so the
# artifact opens dark regardless of the reader's OS preference.
# --------------------------------------------------------------------------
CHROME_CSS = """/* ---- forced GitHub dark (wins over prefers-color-scheme) ---- */
html { background: #0d1117; color-scheme: dark; }
html, body { background: #0d1117; }
:root, .markdown-body {
  color-scheme: dark;
  --fgColor-default: #c9d1d9;
  --fgColor-muted: #8b949e;
  --fgColor-accent: #58a6ff;
  --bgColor-default: #0d1117;
  --bgColor-muted: #161b22;
  --bgColor-neutral-muted: #161b22;
  --borderColor-default: #30363d;
  --borderColor-muted: #30363d;
  --borderColor-neutral-muted: #30363d;
  --borderColor-accent-emphasis: #58a6ff;
  --focus-outlineColor: #58a6ff;
  --fgColor-danger: #f85149;
}
.markdown-body h1, .markdown-body h2, .markdown-body h3,
.markdown-body h4, .markdown-body h5, .markdown-body h6 { color: #f0f6fc; }
.markdown-body table thead tr { background-color: #161b22; }
.markdown-body .highlight, .highlight { background: #161b22; }
/* pygments github-dark on a fixed dark palette */
.highlight .hll { background-color: #6e7681; }
.highlight .c, .highlight .ch, .highlight .cm, .highlight .cp,
.highlight .cpf, .highlight .c1, .highlight .cs { color: #8b949e; font-style: italic; }
.highlight .err { border: 1px solid #f85149; }
.highlight .k, .highlight .kc, .highlight .kd, .highlight .kn,
.highlight .kp, .highlight .kr, .highlight .kt { color: #ff7b72; font-weight: bold; }
.highlight .o, .highlight .ow { color: #ff7b72; font-weight: bold; }
.highlight .gd { color: #ffdcd7; background-color: #67060c; }
.highlight .ge { font-style: italic; }
.highlight .ges { font-weight: bold; font-style: italic; }
.highlight .gr { color: #ffdcd7; }
.highlight .gh { color: #f0f6fc; font-weight: bold; }
.highlight .gi { color: #aff5b4; background-color: #033a16; }
.highlight .go { color: #8b949e; }
.highlight .gp { color: #8b949e; font-weight: bold; }
.highlight .gs { font-weight: bold; }
.highlight .gu { color: #d2a8ff; font-weight: bold; }
.highlight .gt { color: #ff7b72; }
.highlight .m, .highlight .mb, .highlight .mf, .highlight .mh,
.highlight .mi, .highlight .mo, .highlight .il { color: #79c0ff; }
.highlight .s, .highlight .sa, .highlight .sb, .highlight .sc, .highlight .dl,
.highlight .sd, .highlight .s2, .highlight .se, .highlight .sh,
.highlight .si, .highlight .sx, .highlight .sr, .highlight .s1,
.highlight .ss { color: #a5d6ff; }
.highlight .na { color: #79c0ff; }
.highlight .nb { color: #ffa657; }
.highlight .nc { color: #ffa657; font-weight: bold; }
.highlight .no { color: #79c0ff; }
.highlight .nd { color: #d2a8ff; }
.highlight .ni { color: #ffa657; font-weight: bold; }
.highlight .ne { color: #ff7b72; font-weight: bold; }
.highlight .nf { color: #d2a8ff; }
.highlight .nl { color: #79c0ff; }
.highlight .nn { color: #ff7b72; }
.highlight .nt { color: #7ee787; }
.highlight .nv, .highlight .vc, .highlight .vg, .highlight .vi,
.highlight .vm { color: #ffa657; }
.highlight .w { color: #8b949e; }
.highlight .bp { color: #ffa657; }
.highlight .fm { color: #d2a8ff; }

/* ---- page chrome ---- */
* { box-sizing: border-box; }
body {
  margin: 0;
  background: #0d1117;
  color: #c9d1d9;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
  font-size: 16px;
  line-height: 1.5;
}
.gh-header {
  background: #0d1117;
  border-bottom: 1px solid #30363d;
  position: sticky; top: 0; z-index: 20;
}
.gh-header-inner {
  max-width: 1024px; margin: 0 auto; padding: 14px 24px;
  display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap;
}
.gh-left { display: flex; align-items: center; flex-wrap: wrap; }
.gh-repo {
  font-weight: 600; color: #f0f6fc; font-size: 16px;
  padding: 4px 10px; border: 1px solid #30363d; border-radius: 6px; background: #161b22;
  display: inline-block; white-space: nowrap;
}
.gh-tabs { margin-left: 18px; }
.gh-tab {
  display: inline-block; padding: 8px 12px; color: #8b949e; font-size: 14px; font-weight: 500;
  border-bottom: 2px solid transparent; cursor: default; white-space: nowrap;
}
.gh-tab-open { color: #f0f6fc; font-weight: 600; border-bottom-color: #58a6ff; }
.gh-badge {
  display: inline-block; padding: 3px 10px; border-radius: 999px;
  background: #3d1d20; border: 1px solid #f85149; color: #f85149;
  font-size: 12px; font-weight: 600; letter-spacing: .02em; white-space: nowrap;
}
.gh-owners { color: #8b949e; font-size: 12px; margin-left: 10px; white-space: nowrap; }
.gh-note {
  max-width: 1024px; margin: 0 auto; padding: 6px 24px; font-size: 12px; color: #8b949e;
}
.gh-stage {
  max-width: 1024px; margin: 24px auto 80px; padding: 0 24px;
}
.markdown-body {
  background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
  padding: 24px 40px; max-width: 768px; margin: 0 auto; overflow-wrap: break-word;
}
/* ---- fence header bar (language tab + copy) ---- */
.fence-bar {
  display: flex; align-items: center; justify-content: space-between;
  background: #161b22; border: 1px solid #30363d; border-bottom: none;
  border-radius: 6px 6px 0 0; padding: 6px 12px; font-size: 12px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif;
}
.fence-lang {
  color: #8b949e; font-weight: 600; text-transform: lowercase;
  display: inline-flex; align-items: center; gap: 6px;
}
.fence-lang::before {
  content: ""; width: 0; height: 0;
  border-left: 5px solid #8b949e; border-top: 3px solid transparent; border-bottom: 3px solid transparent;
}
.fence-copy {
  border: 1px solid #30363d; background: #161b22; color: #c9d1d9;
  border-radius: 6px; padding: 2px 10px; font-size: 12px; font-weight: 500; cursor: pointer;
  font-family: inherit;
}
.fence-copy:hover { background: #21262d; }
.markdown-body pre > code.highlight,
.markdown-body pre > code {
  border-radius: 0 0 6px 6px;
}
.highlight > pre { margin: 0 !important; border-radius: 0 0 6px 6px !important; border-top: none; }

/* ---- change highlighting (SOP 1.1): every re-render states what changed ---- */
.chg-banner {
  max-width: 1024px; margin: 14px auto 0; padding: 10px 14px; border-radius: 6px;
  font-size: 13px; font-weight: 600; letter-spacing: .01em;
  border: 1px solid #238636; background: #0f2a17; color: #aff5b4;
}
.chg-banner-none {
  border-color: #30363d; background: #161b22; color: #8b949e; font-weight: 500;
}
.chg-panel { max-width: 1024px; margin: 10px auto 0; }
.chg-panel h3 {
  font-size: 12px; color: #8b949e; font-weight: 600; margin: 0 0 6px;
  text-transform: uppercase; letter-spacing: .04em;
}
.chg-diff {
  font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, monospace;
  font-size: 12px; line-height: 1.45; border: 1px solid #30363d; border-radius: 6px;
  background: #161b22; padding: 8px 0; overflow: auto; max-height: 420px;
}
.chg-diff > div { padding: 1px 12px; white-space: pre-wrap; word-break: break-word; }
.chg-diff .add { background: #033a16; color: #aff5b4; }
.chg-diff .del { background: #67060c; color: #ffdcd7; }
.markdown-body p.chg-add, .markdown-body li.chg-add {
  border-left: 3px solid #238636; background: #0f2a17;
  margin-left: -13px; padding-left: 10px;
}
.markdown-body h1.chg-add, .markdown-body h2.chg-add, .markdown-body h3.chg-add,
.markdown-body h4.chg-add, .markdown-body h5.chg-add, .markdown-body h6.chg-add {
  border-left: 3px solid #238636; margin-left: -13px; padding-left: 10px;
}
.markdown-body tr.chg-add > td, .markdown-body tr.chg-add > th { background: #0f2a17; }
"""

FENCE_JS = """<script>
(function () {
  var heads = document.querySelectorAll('.markdown-body pre');
  heads.forEach(function (pre) {
    var code = pre.querySelector('code');
    if (!code) return;
    var lang = 'text';
    var m = code.className && code.className.match(/language-([a-zA-Z0-9_+-]+)/);
    if (m) lang = m[1];
    var bar = document.createElement('div');
    bar.className = 'fence-bar';
    var langSpan = document.createElement('span');
    langSpan.className = 'fence-lang';
    langSpan.textContent = lang;
    var btn = document.createElement('button');
    btn.className = 'fence-copy';
    btn.textContent = 'Copy';
    btn.type = 'button';
    btn.addEventListener('click', function () {
      var text = code.innerText || code.textContent;
      var done = function () {
        btn.textContent = 'Copied';
        setTimeout(function () { btn.textContent = 'Copy'; }, 1500);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, done);
      } else {
        var ta = document.createElement('textarea');
        ta.value = text; document.body.appendChild(ta); ta.select();
        try { document.execCommand('copy'); } catch (e) {}
        document.body.removeChild(ta); done();
      }
    });
    bar.appendChild(langSpan);
    bar.appendChild(btn);
    pre.parentNode.insertBefore(bar, pre);
  });
})();
</script>"""


# --------------------------------------------------------------------------
# Fenced-code handling: render each ```lang block through pygments ourselves
# so the <code> gets class="language-<lang>" (the fence JS reads it) and the
# python block gets github-dark syntax spans. Text fences stay plain text.
# --------------------------------------------------------------------------
_FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})[ \t]*([\w.+-]*)[ \t]*$")
_RAW_HTML_RE = re.compile(r"<!--[\s\S]*?-->|</?[A-Za-z][^>]*>")
_TAG_RE = re.compile(r"<[^>]+>")
_EVENT_ATTR_RE = re.compile(r"\s+on[a-z0-9_-]+\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+)", re.I)
_URL_ATTR_RE = re.compile(r"(\s(?:href|src)\s*=\s*)(\"|')([^\"']*)(\2)", re.I)


def _escape_raw_html(markdown_text: str) -> str:
    """Keep Markdown text, but make author-supplied HTML inert."""
    return _RAW_HTML_RE.sub(lambda m: _html.escape(m.group(0)), markdown_text)


def _safe_markdown_html(html_text: str) -> str:
    """Remove event attributes and unsafe URL schemes from Markdown output."""
    def clean_tag(match):
        tag = _EVENT_ATTR_RE.sub("", match.group(0))
        def clean_url(url_match):
            url = url_match.group(3).strip()
            if re.match(r"(?i)^(?:javascript|vbscript|data):", url):
                return url_match.group(1) + url_match.group(2) + "#" + url_match.group(4)
            return url_match.group(0)
        return _URL_ATTR_RE.sub(clean_url, tag)
    return _TAG_RE.sub(clean_tag, html_text)


def _render_fence(code_text: str, lang: str) -> str:
    lang = (lang or "").strip()
    try:
        lexer = get_lexer_by_name(lang) if lang else TextLexer()
    except Exception:
        lexer = TextLexer()
    # nowrap=True: pygments emits only the escaped spans, no <pre>/<code> wrapper
    # we build ourselves so the language class lands exactly as we want.
    formatter = HtmlFormatter(cssclass="highlight", linenos=False, nowrap=True)
    inner = highlight(code_text + "\n", lexer, formatter)
    lang_class = f" language-{_html.escape(lang)}" if lang else ""
    return (
        '<div class="highlight"><pre><span></span>'
        f'<code class="highlight{lang_class}">{inner}</code></pre></div>'
    )


def split_fences(text: str):
    """Yield ('md', text_chunk) and ('fence', lang, code_text) preserving order."""
    lines = text.split("\n")
    buf: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        m = _FENCE_RE.match(lines[i])
        if not m:
            buf.append(lines[i]); i += 1; continue
        marker_len = len(m.group(1))
        fchar = re.escape(m.group(1)[0])
        close_re = re.compile(rf"^[ \t]*{fchar}{{{marker_len},}}[ \t]*$")
        lang = m.group(2)
        if buf:
            yield ("md", "\n".join(buf)); buf = []
        i += 1
        code: list[str] = []
        while i < n and not close_re.match(lines[i]):
            code.append(lines[i]); i += 1
        if i < n:
            i += 1  # consume closing fence
        yield ("fence", lang, "\n".join(code))
    if buf:
        yield ("md", "\n".join(buf))


def render_markdown(src: str) -> str:
    """Render draft markdown GitHub-style with pygments fences + tables."""
    if not has_pygments:
        raise SystemExit("error: python `markdown` and `pygments` must be installed in the interpreter")
    chunks = []
    for kind, *rest in split_fences(src):
        if kind == "md":
            html_part = _md.markdown(
                _escape_raw_html(rest[0]),
                extensions=["tables", "sane_lists"],
            )
            chunks.append(_safe_markdown_html(html_part))
        else:
            lang, code = rest
            chunks.append(_render_fence(code, lang))
    return "\n".join(chunks)


# --------------------------------------------------------------------------
# SOP 1.1 change highlighting: a re-render must state what changed since the
# previous revision. Baseline sources, in order: --diff-base, a previously
# rendered artifact (--from-html-baseline), the --rev-dir snapshot of the last
# render. With no baseline the banner says so instead of implying "no change".
# --------------------------------------------------------------------------
_MD_BODY_RE = re.compile(r'<div class="markdown-body">(.*)</div>\s*<script', re.S)
_RECOVER_RE = re.compile(r"<(h[1-6]|p|li|tr|blockquote)(?:\s[^>]*)?>(.*?)</\1>", re.S)
_BLOCK_RE = re.compile(r"<(h[1-6]|p|li|tr)((?:\s[^>]*)?)>(.*?)</\1>", re.S)
_TRANSFORMED_RE = re.compile(r"^\|[-\s:|]+\|$")


def _plain(inner: str) -> str:
    """Visible text of an HTML fragment, whitespace-collapsed."""
    return re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", inner))).strip()


def recover_baseline_from_html(html_text: str) -> str:
    """Recover an approximate markdown baseline from a previous rendered artifact.

    Best effort by design: headings, paragraphs, list items and table rows are
    reconstructed, inline markup is flattened. Used to diff a re-render against
    an artifact whose source no longer exists.
    """
    m = _MD_BODY_RE.search(html_text)
    body = m.group(1) if m else html_text
    out: list[str] = []
    for tag, inner in _RECOVER_RE.findall(body):
        if tag.startswith("h"):
            text = _plain(inner)
            if text:
                out.append("#" * int(tag[1]) + " " + text)
        elif tag == "tr":
            cells = [_plain(c) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", inner, re.S)]
            if cells:
                out.append("| " + " | ".join(cells) + " |")
        elif tag == "li":
            text = _plain(inner)
            if text:
                out.append("- " + text)
        elif tag == "blockquote":
            text = _plain(inner)
            if text and not text.startswith("Automated posting"):
                out.append("> " + text)
        else:
            text = _plain(inner)
            if text and not text.startswith("Automated posting"):
                out.append(text)
    # HTML rendering collapses markdown paragraph/list whitespace. The exact
    # source is the authoritative baseline when available; this recovery is only
    # for content comparison, so remove blank layout and renderer-only footer.
    return "\n".join(line for line in out if line.strip()) + "\n"


def diff_source(baseline: str, current: str):
    """Line-level added/removed lists between two markdown sources.

    Blank lines are ignored: they carry no content, and diffing them would drown
    real changes in layout noise.
    """
    a = [l for l in baseline.splitlines() if l.strip()]
    b = [l for l in current.splitlines() if l.strip()]
    added: list[str] = []
    removed: list[str] = []
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "insert"):
            added.extend(b[j1:j2])
        if tag in ("replace", "delete"):
            removed.extend(a[i1:i2])
    return added, removed


def _sig_of_line(line: str) -> str:
    """Text signature of a markdown line, for locating its rendered block."""
    s = re.sub(r"^\s*(#{1,6}\s*|[-*+]\s*|\d+\.\s*)", "", line)
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = s.replace("**", "").replace("|", " ")
    return re.sub(r"\s+", " ", s).strip()


def mark_added(html_text: str, added_lines: list) -> str:
    """Add class="chg-add" to every rendered block matching an added line.

    Text is never altered, so the artifact stays copy-paste faithful; only the
    class attribute changes. Lines the matcher cannot place still appear in the
    change-detail panel, so nothing is silently lost.
    """
    sigs = [s for s in (_sig_of_line(l) for l in added_lines) if s and not _TRANSFORMED_RE.match(s)]
    if not sigs:
        return html_text

    def _repl(m):
        tag, attrs, inner = m.group(1), m.group(2), m.group(3)
        plain = _plain(inner)
        if any(s in plain for s in sigs):
            if "class=" in attrs:
                attrs = attrs.replace('class="', 'class="chg-add ', 1)
            else:
                attrs = attrs + ' class="chg-add"'
            return f"<{tag}{attrs}>{inner}</{tag}>"
        return m.group(0)

    return _BLOCK_RE.sub(_repl, html_text)


def build_change_html(added: list, removed: list, label: str, has_baseline: bool) -> str:
    """Banner plus diff detail. Never silent: states the baseline and the counts."""
    if not has_baseline:
        return ('<div class="chg-banner chg-banner-none">No baseline found for this draft, '
                'so this render carries no change highlighting. Pass --diff-base or --rev-dir.</div>')
    if not added and not removed:
        return (f'<div class="chg-banner chg-banner-none">No content change since '
                f'{_html.escape(label)}: this artifact is identical to the previous revision.</div>')
    parts = [f'<div class="chg-banner">Changed since {_html.escape(label)} &#183; '
             f'{len(added)} line(s) added &#183; {len(removed)} removed</div>']
    rows = ['<div class="chg-panel"><h3>Change detail</h3><div class="chg-diff">']
    for l in removed:
        rows.append(f'<div class="del">{_html.escape("- " + l)}</div>')
    for l in added:
        rows.append(f'<div class="add">{_html.escape("+ " + l)}</div>')
    rows.append("</div></div>")
    return "\n".join(parts + rows)


def _resolve_baseline(args, in_path: Path, draft: str):
    """Baseline markdown for change highlighting, with its human label."""
    if getattr(args, "diff_base", None):
        p = Path(args.diff_base)
        if not p.is_file():
            raise SystemExit(f"error: --diff-base not found: {p}")
        return p.read_text(encoding="utf-8"), args.baseline_label
    if getattr(args, "html_baseline", None):
        p = Path(args.html_baseline)
        if not p.is_file():
            raise SystemExit(f"error: --from-html-baseline not found: {p}")
        return recover_baseline_from_html(p.read_text(encoding="utf-8")), args.baseline_label
    if getattr(args, "rev_dir", None):
        prev = Path(args.rev_dir) / (in_path.stem + ".prev.md")
        if prev.is_file():
            return prev.read_text(encoding="utf-8"), args.baseline_label
    return None, args.baseline_label


def build_page(*, draft: str, repo: str, tab: str, title: str,
               badge: str, src_lines: int, change_html: str = "",
               added_lines: list | None = None,
               owners: str = DEFAULT_OWNERS, state: str = DEFAULT_STATE) -> str:
    markdown_html = render_markdown(draft)
    if added_lines:
        markdown_html = mark_added(markdown_html, added_lines)

    def _esc(s: str) -> str:
        return _html.escape(s, quote=True)

    css = GITHUB_MARKDOWN_CSS + "\n" + CHROME_CSS
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<title>{_esc(title)}</title>
<style>
{css}
</style>
</head>
<body>

<header class="gh-header">
  <div class="gh-header-inner">
    <div class="gh-left">
      <span class="gh-repo">{_esc(repo)}</span>
      <span class="gh-tabs">
        <span class="gh-tab">Issues</span>
        <span class="gh-tab gh-tab-open">{_esc(tab)}</span>
        <span class="gh-tab">Pull requests</span>
        <span class="gh-tab">Projects</span>
      </span>
    </div>
    <div class="gh-right">
      <span class="gh-badge">{_esc(badge)}</span>
      <span class="gh-owners">{_esc(owners)}</span>
    </div>
  </div>
</header>
<div class="gh-note">State: {_esc(state)} &#183; {src_lines} lines of source markdown</div>
{change_html}

<div class="gh-stage">
  <div class="markdown-body">
{markdown_html}
  </div>
</div>
{FENCE_JS}
</body>
</html>
"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="render_draft_html.py",
        description="Render a GitHub draft (.md) as a self-contained GitHub-dark HTML for review.",
    )
    p.add_argument("--in", dest="in_path", required=True, metavar="<draft.md>")
    p.add_argument("--out", dest="out_path", required=True, metavar="<out.html>")
    p.add_argument("--repo", required=True)
    p.add_argument("--tab", required=True, choices=["issue draft", "pull request draft"])
    p.add_argument("--title", required=True)
    p.add_argument("--badge", required=True)
    p.add_argument("--owners", default=DEFAULT_OWNERS,
                   help="header attribution: external drafts pass the poster's handle")
    p.add_argument("--state", default=DEFAULT_STATE, help="header state line")
    p.add_argument("--open", action="store_true", help="open the result in the browser after writing")
    p.add_argument("--diff-base", dest="diff_base", metavar="<prev.md>",
                   help="SOP 1.1: baseline markdown; lines changed against it are highlighted")
    p.add_argument("--from-html-baseline", dest="html_baseline", metavar="<prev.html>",
                   help="SOP 1.1: recover the baseline from a previous render of this draft")
    p.add_argument("--baseline-label", dest="baseline_label", default="the previous revision",
                   help="human label for the baseline shown in the change banner")
    p.add_argument("--rev-dir", dest="rev_dir", metavar="<dir>",
                   help="SOP 1.1: keep <dir>/<slug>.prev.md so every re-render diffs automatically")
    args = p.parse_args(argv)

    in_path = Path(args.in_path)
    if not in_path.is_file():
        p.error(f"input not found: {in_path}")
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    draft = in_path.read_text(encoding="utf-8")
    src_lines = draft.count("\n") + 1 if draft else 0

    baseline_text, baseline_label = _resolve_baseline(args, in_path, draft)
    if baseline_text is None:
        added, removed = [], []
    else:
        added, removed = diff_source(baseline_text, draft)
    change_html = build_change_html(added, removed, baseline_label, baseline_text is not None)

    html = build_page(
        draft=draft, repo=args.repo, tab=args.tab, title=args.title,
        badge=args.badge, src_lines=src_lines, change_html=change_html,
        added_lines=added, owners=args.owners, state=args.state,
    )
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path} ({len(html)} bytes, {src_lines} source lines; "
          f"{len(added)} added / {len(removed)} removed vs {baseline_label})")

    if args.rev_dir:
        rev_dir = Path(args.rev_dir)
        rev_dir.mkdir(parents=True, exist_ok=True)
        (rev_dir / (in_path.stem + ".prev.md")).write_text(draft, encoding="utf-8")

    if args.open:
        webbrowser.open(out_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())