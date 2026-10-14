"""Incremental iLink Markdown compatibility (Tencent openclaw-weixin 2.4.9).

Keep ambiguous syntax across deltas, including code spans which must stay literal.

Adapted from Tencent's openclaw-weixin Markdown filter, licensed under MIT.
Copyright (C) 2026 Tencent. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from __future__ import annotations

import re

_CJK = re.compile(r"[\u2e80-\u9fff\uac00-\ud7af\uf900-\ufaff]")
_TRIGGER = re.compile(r"[\n!*_`]")


class StreamingMarkdownFilter:
    def __init__(self):
        self.buf = ""
        self.sol = True
        self.fence = False
        self.inline = None

    def feed(self, delta: str) -> str:
        self.buf += delta
        return self._pump(False)

    def flush(self) -> str:
        return self._pump(True)

    def _pump(self, eof: bool) -> str:
        out = []
        while self.buf or (eof and self.inline):
            before = (self.buf, self.sol, self.fence, self.inline)
            if self.fence:
                out.append(self._fenced(eof))
            elif self.inline:
                out.append(self._inline(eof))
            elif self.sol:
                out.append(self._line_start(eof))
            else:
                out.append(self._body(eof))
            if before == (self.buf, self.sol, self.fence, self.inline):
                break
        if eof and self.inline:
            marker, acc = self.inline
            out.append(marker + acc)
            self.inline = None
        return "".join(out)

    def _take(self, count: int) -> str:
        value, self.buf = self.buf[:count], self.buf[count:]
        return value

    def _fenced(self, eof: bool) -> str:
        if self.sol:
            if len(self.buf) < 3 and not eof:
                return ""
            if self.buf.startswith("```"):
                nl = self.buf.find("\n")
                if nl < 0 and not eof:
                    return ""
                self.fence = False
                return self._take(nl + 1 if nl >= 0 else len(self.buf))
            self.sol = False
        nl = self.buf.find("\n")
        self.sol = nl >= 0
        return self._take(nl + 1 if nl >= 0 else len(self.buf))

    def _line_start(self, eof: bool) -> str:
        b = self.buf
        if b[0] == "\n":
            return self._take(1)
        if b[0] == "`":
            return self._fence_start(eof)
        if b[0] == "#":
            count = len(b) - len(b.lstrip("#"))
            if count == len(b) and not eof:
                return ""
            if count in (5, 6) and b[count:count + 1] == " ":
                self._take(count + 1)
        elif b[0] in "-*_":
            match = re.match(r"[" + re.escape(b[0]) + r" ]+", b)
            end = match.end()
            if end == len(b) and not eof:
                return ""
            if (end == len(b) or b[end] == "\n") and b[:end].count(b[0]) >= 3:
                return self._take(end + (end < len(b)))
        elif b.isspace() and "\n" not in b and not eof:
            return ""
        self.sol = False
        return ""

    def _fence_start(self, eof: bool) -> str:
        if len(self.buf) < 3 and not eof:
            return ""
        if self.buf.startswith("```"):
            nl = self.buf.find("\n")
            if nl < 0 and not eof:
                return ""
            self.fence = nl >= 0
            return self._take(nl + 1 if nl >= 0 else len(self.buf))
        self.sol = False
        return ""

    def _body(self, eof: bool) -> str:
        match = _TRIGGER.search(self.buf)
        if not match:
            return self._take(len(self.buf))
        if match.start():
            return self._take(match.start())
        char = self.buf[0]
        if char == "\n":
            self.sol = True
            return self._take(1)
        if char == "!":
            if len(self.buf) == 1 and not eof:
                return ""
            if self.buf.startswith("!["):
                self._take(2)
                self.inline = ("![", "")
                return ""
            return self._take(1)
        return self._marker_start(char, eof)

    def _marker_start(self, char: str, eof: bool) -> str:
        count = len(self.buf) - len(self.buf.lstrip(char))
        if count == len(self.buf) and not eof:
            return ""
        if char == "`":
            marker = self._take(count)
        elif count >= 3:
            marker = self._take(3)
        elif count == 2:
            return self._take(2)
        elif self.buf[1:2] in ("", " ", "\n"):
            return self._take(1)
        else:
            marker = self._take(1)
        self.inline = (marker, "")
        return ""

    def _inline(self, eof: bool) -> str:
        marker, acc = self.inline
        acc += self._take(len(self.buf))
        self.inline = (marker, acc)
        if marker == "![":
            return self._image(acc)
        end = self._closing_marker(acc, marker, eof)
        nl = acc.find("\n") if len(marker) == 1 and marker != "`" else -1
        if nl >= 0 and (end < 0 or nl < end):
            self.inline = None
            self.buf = acc[nl + 1:]
            self.sol = True
            return marker + acc[:nl + 1]
        if end < 0:
            return ""
        content = acc[:end]
        self.buf = acc[end + len(marker):]
        self.inline = None
        if marker[0] in "*_" and _CJK.search(content):
            return content
        return marker + content + marker

    @staticmethod
    def _closing_marker(acc: str, marker: str, eof: bool) -> int:
        if len(marker) > 1 or marker == "`":
            return acc.find(marker)
        for match in re.finditer(re.escape(marker) + "+", acc):
            if match.end() == len(acc) and not eof:
                return -1
            if len(match.group()) == 1:
                return match.start()
        return -1

    def _image(self, acc: str) -> str:
        cb = acc.find("]")
        if cb < 0 or cb + 1 >= len(acc):
            return ""
        if acc[cb + 1] != "(":
            self.inline = None
            self.buf = acc[cb + 1:]
            return "![" + acc[:cb + 1]
        cp = acc.find(")", cb + 2)
        if cp >= 0:
            self.inline = None
            self.buf = acc[cp + 1:]
        return ""


def filter_markdown(text: str) -> str:
    parser = StreamingMarkdownFilter()
    return parser.feed(text) + parser.flush()
