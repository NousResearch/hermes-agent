/**
 * PTY resume output sanitizer — strips pathological ANSI sequences that
 * Ink's two-pass virtual scroll emits during session resume.
 *
 * Three properties of the real pipeline drive this design:
 *
 * 1. **The PTY is in cooked mode (ONLCR).** `hermes_cli/pty_bridge.py` spawns
 *    via `ptyprocess.PtyProcess.spawn()` and never calls `setraw()`, so the
 *    line discipline rewrites every LF the child writes as CRLF. A burst
 *    therefore arrives as `\r\n\r\n\r\n…`, never as bare `\n\n\n…`, and any
 *    filter matching `\n{50,}` would never fire in production.
 *
 * 2. **Reads are chunked, not message-framed.** `bridge.read()` does
 *    `os.read(fd, 65536)` per drain tick and `pty_session.py` forwards each
 *    read as its own binary WebSocket frame. A CSI escape, a UTF-8 code
 *    point, *and* a long blank-line run can each straddle a frame boundary,
 *    so all three need trailing-state buffering to survive reassembly.
 *
 * 3. **Erase codes are only pathological during the resume replay.** Once the
 *    replay has settled, `ESC[K` / `ESC[X` are exactly how a TUI clears stale
 *    glyphs for spinners, progress bars, and status lines. Stripping them
 *    forever corrupts normal interactive output, so suppression is bounded to
 *    a short window after connect (see PTY_RESUME_SANITIZE_WINDOW_MS).
 */

/** A blank-line run: CRLF (real PTY, cooked mode) or bare LF (raw-mode PTY). */
const BLANK_LINE_BURST = /(?:\r?\n){50,}/g;
// eslint-disable-next-line no-control-regex -- intentional ESC byte in ANSI sequence parser
const ERASE_LINE = /\x1b\[\d*K/g;
// eslint-disable-next-line no-control-regex -- intentional ESC byte in ANSI sequence parser
const ERASE_CHAR = /\x1b\[\d*X/g;

/** Still-incomplete trailing escape: "\x1b", "\x1b[", "\x1b[\d*". */
// eslint-disable-next-line no-control-regex -- intentional ESC byte in ANSI sequence parser
const PARTIAL_ESC = /^\x1b(?:\[\d*)?$/;

/**
 * A trailing run of newlines that may continue into the next frame. Held back
 * so a burst split across frames still meets the collapse threshold instead of
 * slipping through as sub-threshold fragments. A lone trailing `\r` is
 * included: a frame boundary can fall between the CR and LF of a CRLF pair,
 * and emitting the CR early would break one run into two shorter ones.
 *
 * Always matches (possibly empty, at end of input).
 */
const TRAILING_NEWLINES = /(?:\r?\n)*\r?$/;

/** Collapsed form of a pathological burst: one blank row, CRLF for xterm. */
const COLLAPSED_BURST = "\r\n\r\n";

/**
 * Apply all suppression rules to a safely-completed string.
 *
 * @param stripErase When false, `ESC[K` / `ESC[X` are preserved. Blank-line
 *   burst collapsing still applies — a thousand-row burst is pathological
 *   whenever it appears, but erase codes are legitimate once resume settles.
 * Exported for focused unit-testing of filter behaviour.
 */
export function applyPtyFilters(input: string, stripErase = true): string {
  const collapsed = input.replace(BLANK_LINE_BURST, COLLAPSED_BURST);
  if (!stripErase) return collapsed;
  return collapsed.replace(ERASE_LINE, "").replace(ERASE_CHAR, "");
}

/** Stateful chunk processor that guards against cross-frame split sequences. */
export class PtyResumeSanitizer {
  #pending = "";
  #stripErase = true;

  /**
   * Stop stripping erase codes while continuing to collapse blank-line bursts.
   * Called when the resume-replay window closes so that ordinary interactive
   * redraws (spinners, progress bars, status lines) keep their `ESC[K`.
   */
  endEraseSuppression(): void {
    this.#stripErase = false;
  }

  /** True while erase-code stripping is still active. */
  get isSuppressingErase(): boolean {
    return this.#stripErase;
  }

  /** Feed one decoded WebSocket frame payload. Returns the sanitized output. */
  next(chunk: string): string {
    const combined = this.#pending + chunk;
    if (combined === "") {
      this.#pending = "";
      return "";
    }

    // Hold back a trailing partial escape so a CSI split across frames is
    // still recognised once its terminating byte arrives.
    const lastEsc = combined.lastIndexOf("\x1b");
    if (lastEsc !== -1 && PARTIAL_ESC.test(combined.slice(lastEsc))) {
      this.#pending = combined.slice(lastEsc);
      return applyPtyFilters(combined.slice(0, lastEsc), this.#stripErase);
    }

    // Hold back a trailing newline run so a burst spanning frames accumulates
    // to the collapse threshold instead of leaking through in fragments.
    // TRAILING_NEWLINES always matches; an empty match at end of input means
    // the frame does not end mid-run and nothing needs to be held back.
    const trailing = TRAILING_NEWLINES.exec(combined) as RegExpExecArray;
    if (trailing.index === 0) {
      // The whole frame is newlines — keep accumulating, emit nothing yet.
      this.#pending = combined;
      return "";
    }
    this.#pending = trailing[0];
    return applyPtyFilters(combined.slice(0, trailing.index), this.#stripErase);
  }

  /**
   * Drain buffered state at end of stream.
   *
   * A buffered *partial escape* is dropped rather than written: `#pending` only
   * ever holds an incomplete sequence, and emitting one would leave xterm's
   * parser in an "in-escape" state that swallows subsequent output after
   * reconnect. A buffered *newline run* is safe and is emitted (collapsed).
   */
  flush(): string {
    const last = this.#pending;
    this.#pending = "";
    if (last === "" || last.includes("\x1b")) return "";
    return applyPtyFilters(last, this.#stripErase);
  }
}
