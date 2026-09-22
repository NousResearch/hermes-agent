<!doctype html>
<!--
  device-diag: real-device scrub diagnostic for scrollcraft builds.

  Headless verification cannot reproduce a phone's video decoder, autoplay
  policy, Low Power Mode, or touch scrolling. When a clip is reported frozen
  on a real device, deploy this page beside the site (same origin as the
  clips), open it on the device, scroll up and down a few times, and read the
  verdicts. One screenshot isolates the failing layer:

    suspect-blob FROZEN, suspect-direct MOVING  -> the blob: URL path is the problem
    suspect FROZEN, known-good MOVING           -> something specific to that clip's lifecycle
    all MOVING                                  -> the video is fine; the fault is in the
                                                   engine's reveal/tick path on the real page
    all FROZEN                                  -> the device refuses scrub decode; use the
                                                   poster fallback for that act

  To use: edit the TESTS array below. Point the first two entries at the
  suspect clip (same file, loaded two ways) and the third at a clip known to
  work on the device. Paths resolve against this page's URL.

  Scrubbing is driven by a rAF poll of scrollY, not scroll events, because a
  driven browser can move the page without firing them.
-->
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>scrub diagnostic</title>
<style>
  * { margin: 0; box-sizing: border-box; }
  html { background: #0b0f14; color: #e8edf2; font: 13px/1.45 ui-monospace, Menlo, Consolas, monospace; }
  body { height: 520vh; }
  .panel { position: fixed; inset: 0; padding: 10px; display: flex; flex-direction: column; gap: 8px; }
  h1 { font-size: 14px; font-weight: 600; }
  .hint { color: #8fa0b3; }
  .row { display: flex; gap: 8px; flex: 0 0 auto; }
  .cell { flex: 1 1 0; min-width: 0; }
  .cell video { width: 100%; aspect-ratio: 9 / 16; object-fit: cover; background: #000; display: block; border: 1px solid #26303c; border-radius: 6px; }
  .name { font-weight: 600; margin: 6px 0 2px; }
  .verdict { font-size: 15px; font-weight: 700; padding: 2px 6px; border-radius: 4px; display: inline-block; }
  .wait { background: #26303c; }
  .ok { background: #0f5132; color: #b6f0cd; }
  .bad { background: #6b1220; color: #ffc2cd; }
  .stats { color: #aab7c4; font-size: 11.5px; white-space: pre-line; }
  #meta { color: #8fa0b3; font-size: 11.5px; }
  #log { flex: 1 1 auto; overflow: auto; background: #10161d; border: 1px solid #26303c; border-radius: 6px; padding: 8px; font-size: 11px; color: #9fb4c8; white-space: pre-wrap; }
</style>
</head>
<body>
<div class="panel">
  <h1>scrub diagnostic — SCROLL UP AND DOWN A FEW TIMES, then read the verdicts</h1>
  <div class="hint">A: suspect clip, blob URL (how the engine loads it) &nbsp; B: suspect clip, direct file &nbsp; C: a clip that works on this device, blob URL</div>
  <div class="row" id="cells"></div>
  <div id="meta"></div>
  <div id="log"></div>
</div>
<script>
(function () {
  'use strict';

  // EDIT THESE for the build under test.
  var SUSPECT = 'assets/01-suspect-m.mp4';
  var KNOWN_GOOD = 'assets/02-known-good-m.mp4';
  var TESTS = [
    { key: 'A', label: 'A suspect blob', src: SUSPECT, mode: 'blob' },
    { key: 'B', label: 'B suspect direct', src: SUSPECT, mode: 'direct' },
    { key: 'C', label: 'C known-good blob', src: KNOWN_GOOD, mode: 'blob' }
  ];

  var logEl = document.getElementById('log');
  function log(m) {
    logEl.textContent += m + '\n';
    logEl.scrollTop = logEl.scrollHeight;
  }
  window.addEventListener('error', function (e) { log('JS ERROR: ' + e.message); });

  var maxScrolled = 0;
  var cellsEl = document.getElementById('cells');
  var canvas = document.createElement('canvas');
  canvas.width = 16; canvas.height = 16;
  var ctx = canvas.getContext('2d', { willReadFrequently: true });

  function make(t) {
    var cell = document.createElement('div');
    cell.className = 'cell';
    cell.innerHTML = '<div class="name">' + t.label + '</div>' +
      '<span class="verdict wait" id="v' + t.key + '">…</span>' +
      '<video muted playsinline preload="auto" id="el' + t.key + '"></video>' +
      '<div class="stats" id="s' + t.key + '"></div>';
    cellsEl.appendChild(cell);

    var T = {
      key: t.key, el: cell.querySelector('video'),
      verdict: cell.querySelector('.verdict'),
      stats: cell.querySelector('.stats'),
      seeked: 0, hashes: {}, hashCount: 0, prime: 'not tried',
      primed: false, priming: false, err: ''
    };
    T.el.muted = true; T.el.playsInline = true;

    T.el.addEventListener('seeked', function () { T.seeked++; });
    T.el.addEventListener('error', function () {
      var e = T.el.error;
      T.err = 'MediaError code ' + (e ? e.code : '?');
      log(t.key + ': ' + T.err);
    });
    T.el.addEventListener('loadedmetadata', function () {
      log(t.key + ': metadata, duration ' + T.el.duration.toFixed(2) + 's');
      try { T.el.currentTime = 0.001; } catch (e) {}
      prime(T, 'load'); // same as the engine: try with no gesture first
    });

    if (t.mode === 'blob') {
      fetch(t.src)
        .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.blob(); })
        .then(function (b) {
          log(t.key + ': fetched ' + (b.size / 1048576).toFixed(2) + ' MB, blob URL set');
          T.el.src = URL.createObjectURL(b);
        })
        .catch(function (e) { T.err = 'fetch failed: ' + e.message; log(t.key + ': ' + T.err); });
    } else {
      T.el.src = t.src;
      log(t.key + ': direct src set');
    }
    return T;
  }

  function prime(T, why) {
    if (T.primed || T.priming || !T.el.src) return;
    T.priming = true;
    var pr;
    try { pr = T.el.play(); } catch (e) { T.priming = false; T.prime = 'threw (' + why + ')'; return; }
    setTimeout(function () {
      if (T.priming) { T.priming = false; if (T.prime === 'not tried') T.prime = 'play() never settled (' + why + ')'; }
    }, 2000);
    if (pr && pr.then) {
      pr.then(function () {
        T.priming = false; T.primed = true; T.prime = 'OK (' + why + ')';
        try { T.el.pause(); } catch (e) {}
        log(T.key + ': primed via ' + why);
      }, function (e) {
        T.priming = false; T.prime = (e && e.name ? e.name : 'rejected') + ' (' + why + ')';
        log(T.key + ': play() rejected via ' + why + ': ' + (e && e.name));
      });
    } else { T.priming = false; T.primed = true; T.prime = 'OK sync (' + why + ')'; try { T.el.pause(); } catch (e) {} }
  }

  var tests = TESTS.map(make);

  function primeAll(ev) { tests.forEach(function (T) { prime(T, ev); }); }
  ['touchstart', 'touchend', 'pointerdown', 'click', 'scroll'].forEach(function (ev) {
    addEventListener(ev, function () { primeAll(ev); }, { passive: true });
  });

  // scroll position drives every clip, exactly like the engine. Polled from a
  // rAF loop rather than a scroll listener, so the diagnostic cannot be blinded
  // by an environment that moves the page without firing scroll events.
  var lastY = -1, lastPrimeTry = 0;
  function loop() {
    var maxY = document.body.scrollHeight - innerHeight;
    var p = maxY > 0 ? Math.max(0, Math.min(1, scrollY / maxY)) : 0;
    if (scrollY !== lastY) {
      if (lastY !== -1 && performance.now() - lastPrimeTry > 600) {
        lastPrimeTry = performance.now();
        primeAll('scrollmove');
      }
      lastY = scrollY;
      maxScrolled = Math.max(maxScrolled, p);
    }
    tests.forEach(function (T) {
      if (!T.el.duration || T.el.seeking) return;
      var t = p * T.el.duration * 0.98;
      if (Math.abs(T.el.currentTime - t) > 0.02) {
        try { T.el.currentTime = t; } catch (e) {}
      }
    });
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);

  function frameHash(T) {
    if (T.el.readyState < 2) return null;
    try {
      ctx.drawImage(T.el, 0, 0, 16, 16);
      var d = ctx.getImageData(0, 0, 16, 16).data, h = 0;
      for (var i = 0; i < d.length; i += 7) h = (h * 31 + d[i]) >>> 0;
      return h;
    } catch (e) { return null; }
  }

  setInterval(function () {
    tests.forEach(function (T) {
      var h = frameHash(T);
      if (h !== null && !T.hashes[h]) { T.hashes[h] = 1; T.hashCount++; }
      T.stats.textContent =
        'prime: ' + T.prime +
        '\nreadyState ' + T.el.readyState + ' · seeks done ' + T.seeked +
        '\ntime ' + T.el.currentTime.toFixed(2) + 's · frames seen ' + T.hashCount +
        (T.err ? '\n' + T.err : '');
      if (T.hashCount >= 3) { T.verdict.textContent = 'MOVING'; T.verdict.className = 'verdict ok'; }
      else if (T.err) { T.verdict.textContent = 'ERROR'; T.verdict.className = 'verdict bad'; }
      else if (maxScrolled > 0.35) { T.verdict.textContent = 'FROZEN'; T.verdict.className = 'verdict bad'; }
    });
  }, 250);

  document.getElementById('meta').textContent =
    navigator.userAgent + ' · ' + innerWidth + 'x' + innerHeight +
    ' · dpr ' + devicePixelRatio;
})();
</script>
</body>
</html>
