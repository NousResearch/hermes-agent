/*
 * Clone of ui.html for systems without chromium.
 *
 *   /usr/bin/osascript -l JavaScript update-panel.js <status-file>
 *
 * Must remain here, next to posix.sh.
 * The desktop spawns this directory as-is (see resolvePosixScriptHandoff) and the script references it via
 * its own path.
 */
ObjC.import('AppKit')
ObjC.import('QuartzCore') // CACurrentMediaTime

// ── Fourier Flow — constants and math ported verbatim from ui.html ──────────
// Keep in sync with ui.html / apps/desktop/src/components/ui/loader.tsx:
// durationMs 2200, particleCount 92, pulseDurationMs 2000, strokeWidth 4.2,
// trailSpan 0.31, PATH_STEPS 240, viewBox 0 0 100 100 (center 50,50).
const TWO_PI = Math.PI * 2

const CURVE = {
  durationMs: 2200,
  particleCount: 92,
  pulseDurationMs: 2000,
  strokeWidth: 4.2,
  trailSpan: 0.31,
  point(progress, detailScale) {
    const t = progress * TWO_PI
    const mix = 1 + detailScale * 0.16
    const x = 17 * Math.cos(t) + 7.5 * Math.cos(3 * t + 0.6 * mix) + 3.2 * Math.sin(5 * t - 0.4)
    const y = 15 * Math.sin(t) + 8.2 * Math.sin(2 * t + 0.25) - 4.2 * Math.cos(4 * t - 0.5 * mix)

    return { x: 50 + x, y: 50 + y }
  }
}

const PATH_STEPS = 240
const norm = progress => ((progress % 1) + 1) % 1

function detailScaleFor (time, phaseOffset) {
  const p = ((time + phaseOffset * CURVE.pulseDurationMs) % CURVE.pulseDurationMs) / CURVE.pulseDurationMs

  return 0.52 + ((Math.sin(p * TWO_PI + 0.55) + 1) / 2) * 0.48
}

function particleFor (index, progress, detailScale) {
  const tail = index / (CURVE.particleCount - 1)
  const { x, y } = CURVE.point(norm(progress - tail * CURVE.trailSpan), detailScale)
  const fade = (1 - tail) ** 0.56

  return { x, y, opacity: 0.04 + fade * 0.96, radius: 0.9 + fade * 2.7 }
}

// Render one animation frame into an 80x80 NSImage (the JS analog of the
// page's <svg> repaint). AppKit is bottom-left origin; the curve math is
// top-left like SVG, so flip Y around the 50,50 center — the math itself is
// untouched.
function renderLoaderFrame (timeMs, phaseOffset, fg) {
  const size = 80
  const img = $.NSImage.alloc.initWithSize($.NSMakeSize(size, size))
  img.lockFocus

  const progress = norm(timeMs / CURVE.durationMs)
  const detailScale = detailScaleFor(timeMs, phaseOffset)

  const ghost = fg.colorWithAlphaComponent(0.1)
  ghost.set
  const path = $.NSBezierPath.bezierPath
  for (let i = 0; i <= PATH_STEPS; i++) {
    const { x, y } = CURVE.point(i / PATH_STEPS, detailScale)
    const px = x / 100 * size
    const py = size - y / 100 * size
    if (i === 0) path.moveToPoint($.NSMakePoint(px, py))
    else path.lineToPoint($.NSMakePoint(px, py))
  }
  path.setLineWidth(CURVE.strokeWidth)
  path.setLineCapStyle($.NSRoundLineCapStyle)
  path.setLineJoinStyle($.NSRoundLineJoinStyle)
  path.stroke

  for (let index = 0; index < CURVE.particleCount; index++) {
    const p = particleFor(index, progress, detailScale)
    const px = p.x / 100 * size
    const py = size - p.y / 100 * size
    fg.colorWithAlphaComponent(p.opacity).set
    $.NSBezierPath.bezierPathWithOvalInRect(
      $.NSMakeRect(px - p.radius, py - p.radius, p.radius * 2, p.radius * 2)
    ).fill
  }

  img.unlockFocus
  return img
}

function isDarkAppearance () {
  const app = $.NSApplication.sharedApplication
  const match = app.effectiveAppearance.bestMatchFromAppearancesWithNames(
    $(['NSAppearanceNameAqua', 'NSAppearanceNameDarkAqua'])
  )
  return ObjC.unwrap(match) === 'NSAppearanceNameDarkAqua'
}

// ui.html: light fg #1a1a1a; dark #d6d6d6 on #232323.
function colors () {
  return isDarkAppearance()
    ? {
        bg: $.NSColor.colorWithCalibratedRedGreenBlueAlpha(35 / 255, 35 / 255, 35 / 255, 1),
        fg: $.NSColor.colorWithCalibratedRedGreenBlueAlpha(214 / 255, 214 / 255, 214 / 255, 1)
      }
    : {
        bg: $.NSColor.whiteColor,
        fg: $.NSColor.colorWithCalibratedRedGreenBlueAlpha(26 / 255, 26 / 255, 26 / 255, 1)
      }
}

function readStatus (path) {
  // write_status emits a flat, one-line JSON pair we control, so a missing
  // or malformed file simply yields undefined (no update this tick). The
  // writer's atomic mv guarantees we never observe a partial file.
  try {
    const data = $.NSString.stringWithContentsOfFileEncodingError(path, $.NSUTF8StringEncoding, null)
    if (data.js === '') return undefined
    const state = JSON.parse(ObjC.unwrap(data))
    if (!state || typeof state.status !== 'string') return undefined
    return state
  } catch (_) {
    return undefined
  }
}

function wrappedLabel (text, font, color, frame) {
  const label = $.NSTextField.wrappingLabelWithString(text)
  label.font = font
  if (color) label.textColor = color
  label.editable = false
  label.bordered = false
  label.drawsBackground = false
  // Alignment must reach the cell: setAlignment on the field alone does not
  // propagate for wrapping labels, and left-aligned text inside a full-width
  // frame reads as "centered box, left-aligned text".
  label.alignment = 1
  label.cell.alignment = 1
  label.frame = frame
  return label
}

function run (argv) {
  if (argv.length < 1) throw new Error('usage: osascript -l JavaScript update-panel.js <status-file>')
  const statusPath = argv[0]

  const app = $.NSApplication.sharedApplication
  // Accessory: no Dock icon — this is a status panel, not an app.
  app.setActivationPolicy($.NSApplicationActivationPolicyAccessory)

  const { bg, fg } = colors()

  // 280x320 like the Chrome --window-size the shim uses.
  const styleMask =
    $.NSWindowStyleMaskTitled |
    $.NSWindowStyleMaskClosable

  const win = $.NSWindow.alloc.initWithContentRectStyleMaskBackingDefer(
    $.NSMakeRect(0, 0, 280, 320),
    styleMask,
    $.NSBackingStoreBuffered,
    false
  )

  // Don't allow closing while Hermes is updating.
  const closeButton = win.standardWindowButton($.NSWindowCloseButton)
  closeButton.enabled = false

  win.title = 'Hermes'
  win.center
  win.releasedWhenClosed = false
  win.backgroundColor = bg
  const content = win.contentView

  // Loader: 80x80, centered horizontally, near the top (AppKit Y-up).
  const loaderView = $.NSImageView.alloc.initWithFrame($.NSMakeRect(100, 226, 80, 80))
  loaderView.imageScaling = $.NSImageScaleProportionallyUpOrDown
  content.addSubview(loaderView)

  const title = wrappedLabel('Updating Hermes', $.NSFont.systemFontOfSize(18), fg,
    $.NSMakeRect(0, 178, 280, 26))
  const line = wrappedLabel('Hermes will open once done.', $.NSFont.systemFontOfSize(12), null,
    $.NSMakeRect(24, 118, 232, 54))
  content.addSubview(title)
  content.addSubview(line)

  win.makeKeyAndOrderFront(null)
  app.activateIgnoringOtherApps(true)

  // Elapsed clock = when the panel started, like serve-ui.py's started_at.
  // Random phase like the page's Math.random() phaseOffset.
  const startedAtMs = $.CACurrentMediaTime() * 1000
  const phaseOffset = Math.random()
  const runloop = $.NSRunLoop.currentRunLoop
  let everPublished = false
  let settled = null // null = running; else 'done' | 'manual' | 'error'
  let message = ''

  while (settled === null) {
    const state = readStatus(statusPath)
    if (state) {
      everPublished = true
      if (state.message) {
        message = String(state.message)
        line.stringValue = message
      }
      if (['done', 'manual', 'error'].includes(state.status)) settled = state.status
    } else if (everPublished) {
      // The status file vanished after having existed: the shim removes it
      // right after publishing a terminal state and tearing down its UI.
      // Waiting forever stranded the panel on its last stage — observed on a
      // real update. A disappearance reads as done; an error publish keeps
      // its file alive through the 15s leave-window grace, so this cannot
      // mask a failure.
      settled = 'done'
    }

    if (settled === null) {
      loaderView.image = renderLoaderFrame($.CACurrentMediaTime() * 1000 - startedAtMs, phaseOffset, fg)
      const screen = $.NSScreen.mainScreen;
      const refreshRate = screen.maximumFramesPerSecond;
      const frameInterval = 1.0 / refreshRate;

      // Pump the main runloop at the monitor's refresh rate.
      runloop.runModeBeforeDate(
        $.NSDefaultRunLoopMode,
        $.NSDate.dateWithTimeIntervalSinceNow(frameInterval)
      );
    }
  }
  
  closeButton.enabled = true

  // Terminal states: swap the loader for the glyph; title/line verbatim
  // from ui.html's apply().
  loaderView.hidden = true
  title.stringValue = settled === 'error' ? 'Failed to update' : 'Update complete'
  if (settled === 'done') {
    line.stringValue = 'Opening Hermes…\nYou can close this window.'
  } else if (settled === 'manual') {
    line.stringValue = message || 'Reopen Hermes to finish.'
  } else {
    line.stringValue = 'Run hermes debug share in a terminal to send a report.'
  }
  const glyph = wrappedLabel(settled === 'error' ? '✕' : '✓',
    $.NSFont.systemFontOfSize(44), fg, $.NSMakeRect(0, 226, 280, 80))
  content.addSubview(glyph)

  if (settled === 'done') {
    // The app is coming back; the shim's job is over.
    linger(1.5)
  } else {
    // manual/error: mirror stop_ui leave-window so the message is readable
    // even without it (the durable result dialog covers the rest).
    linger(15)
  }
}

function linger (seconds) {
  const runloop = $.NSRunLoop.currentRunLoop
  const end = $.CACurrentMediaTime() + seconds

  while ($.CACurrentMediaTime() < end) {
    runloop.runModeBeforeDate(
      $.NSDefaultRunLoopMode,
      $.NSDate.dateWithTimeIntervalSinceNow(0.05)
    )
  }
}
