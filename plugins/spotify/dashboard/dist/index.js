/* Spotify Web Playback dashboard widget — MVP skeleton (issue #15182).
 *
 * Hand-written IIFE that uses the Hermes Plugin SDK globals so the
 * bundle ships without an upstream build step.  All UI is composed
 * with React.createElement; shared layout primitives come from
 * SDK.components (Card / Button / Badge).
 *
 * Loads Spotify's Web Playback SDK from sdk.scdn.co at mount time and
 * registers the browser tab itself as a Connect device named
 * "Hermes Dashboard".  Once registered, any tool call that targets
 * that device by id plays audio in this tab.
 *
 * Backend (plugin_api.py):
 *   GET /api/plugins/spotify/status — login-state probe (no token)
 *   GET /api/plugins/spotify/token  — fresh access token; auto-refreshes
 *
 * Premium-only:
 *   Spotify's SDK fires `account_error` if you try to initialise it on a
 *   Free account.  We intercept that and render a clear message instead
 *   of leaving the widget half-functional.
 *
 * Out of scope for this MVP (open to follow-ups):
 *   - playlist/library browsing
 *   - offline playback (SDK doesn't support it)
 *   - bringing audio over to other tabs / multi-tab device dedup
 */
(function () {
  "use strict";

  const SDK = window.__HERMES_PLUGIN_SDK__;
  if (!SDK || !window.__HERMES_PLUGINS__) return;

  const React = SDK.React;
  const { useState, useEffect, useRef, useCallback } = SDK.hooks;
  const C = SDK.components;
  const cn = (SDK.utils && SDK.utils.cn) || ((...xs) => xs.filter(Boolean).join(" "));
  const fetchJSON = SDK.fetchJSON || ((url, opts) =>
    fetch(url, opts).then((r) => (r.ok ? r.json() : r.json().then((d) => Promise.reject(d))))
  );

  const h = React.createElement;

  const SDK_SCRIPT_ID = "spotify-web-playback-sdk";
  const SDK_SCRIPT_SRC = "https://sdk.scdn.co/spotify-player.js";
  const DEVICE_NAME = "Hermes Dashboard";

  // ---------------------------------------------------------------------------
  // SDK loader — loads once per page, then resolves on every subsequent call.
  // ---------------------------------------------------------------------------

  let _sdkLoadPromise = null;

  function loadSpotifySdk() {
    if (window.Spotify && window.Spotify.Player) return Promise.resolve();
    if (_sdkLoadPromise) return _sdkLoadPromise;

    _sdkLoadPromise = new Promise((resolve, reject) => {
      // The SDK only signals ready by calling
      // window.onSpotifyWebPlaybackSDKReady — that's the documented hook.
      const prevReady = window.onSpotifyWebPlaybackSDKReady;
      window.onSpotifyWebPlaybackSDKReady = function () {
        if (typeof prevReady === "function") {
          try { prevReady(); } catch (_) { /* ignore */ }
        }
        resolve();
      };
      // Some browsers may have the script already injected by another
      // copy of the widget — fall back to the load event in that case.
      const existing = document.getElementById(SDK_SCRIPT_ID);
      if (existing) {
        if (window.Spotify && window.Spotify.Player) resolve();
        existing.addEventListener("load", () => resolve());
        existing.addEventListener("error", () => reject(new Error("Spotify SDK failed to load")));
        return;
      }
      const script = document.createElement("script");
      script.id = SDK_SCRIPT_ID;
      script.src = SDK_SCRIPT_SRC;
      script.async = true;
      script.onerror = () => {
        _sdkLoadPromise = null;
        reject(new Error("Spotify SDK failed to load"));
      };
      document.head.appendChild(script);
    });

    return _sdkLoadPromise;
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  function formatMs(ms) {
    if (!Number.isFinite(ms) || ms < 0) return "0:00";
    const total = Math.floor(ms / 1000);
    const m = Math.floor(total / 60);
    const s = total % 60;
    return m + ":" + (s < 10 ? "0" : "") + s;
  }

  function trackArtists(state) {
    try {
      const t = state && state.track_window && state.track_window.current_track;
      if (!t || !Array.isArray(t.artists)) return "";
      return t.artists.map((a) => a.name).filter(Boolean).join(", ");
    } catch (_) {
      return "";
    }
  }

  function trackName(state) {
    try {
      return (state && state.track_window && state.track_window.current_track && state.track_window.current_track.name) || "";
    } catch (_) {
      return "";
    }
  }

  function trackArtUrl(state) {
    try {
      const images = state && state.track_window && state.track_window.current_track && state.track_window.current_track.album && state.track_window.current_track.album.images;
      if (!Array.isArray(images) || images.length === 0) return null;
      // Prefer ~300px image when available, else the first.
      const mid = images.find((i) => i && i.width >= 240 && i.width <= 360);
      return (mid || images[0]).url || null;
    } catch (_) {
      return null;
    }
  }

  // ---------------------------------------------------------------------------
  // Main component
  // ---------------------------------------------------------------------------

  function SpotifyPage() {
    // phase: "init" | "logged_out" | "loading_sdk" | "ready" | "not_premium" | "error"
    const [phase, setPhase] = useState("init");
    const [errorMessage, setErrorMessage] = useState("");
    const [reloginRequired, setReloginRequired] = useState(false);
    const [deviceId, setDeviceId] = useState(null);
    const [playerState, setPlayerState] = useState(null);
    const [localPositionMs, setLocalPositionMs] = useState(0);
    const [volume, setVolumeState] = useState(0.5);

    const playerRef = useRef(null);
    const lastStateAtRef = useRef(0);
    const lastPositionRef = useRef(0);
    const mountedRef = useRef(true);

    // ---- Bootstrap: check login, load SDK, init Player -----------------------

    const init = useCallback(async () => {
      setPhase("init");
      setErrorMessage("");
      setReloginRequired(false);
      try {
        const status = await fetchJSON("/api/plugins/spotify/status");
        if (!status || !status.logged_in) {
          if (!mountedRef.current) return;
          setPhase("logged_out");
          return;
        }
      } catch (err) {
        if (!mountedRef.current) return;
        setErrorMessage("Could not check Spotify login: " + (err && err.message ? err.message : "unknown error"));
        setPhase("error");
        return;
      }

      if (!mountedRef.current) return;
      setPhase("loading_sdk");
      try {
        await loadSpotifySdk();
      } catch (err) {
        if (!mountedRef.current) return;
        setErrorMessage("Could not load Spotify SDK: " + (err && err.message ? err.message : "unknown error"));
        setPhase("error");
        return;
      }

      if (!mountedRef.current) return;
      if (!window.Spotify || !window.Spotify.Player) {
        setErrorMessage("Spotify SDK loaded but global is missing.");
        setPhase("error");
        return;
      }

      const player = new window.Spotify.Player({
        name: DEVICE_NAME,
        getOAuthToken: (cb) => {
          // Called once on init and again whenever a token expires.
          fetchJSON("/api/plugins/spotify/token")
            .then((data) => {
              if (data && data.access_token) {
                cb(data.access_token);
              } else {
                cb("");
              }
            })
            .catch((err) => {
              if (mountedRef.current) {
                const detail = (err && err.detail) || err;
                setReloginRequired(Boolean(detail && detail.relogin_required));
                setErrorMessage(
                  "Spotify token refresh failed: " +
                  ((detail && detail.message) || (err && err.message) || "unknown")
                );
                setPhase("error");
              }
              cb("");
            });
        },
        volume: 0.5,
      });

      player.addListener("ready", ({ device_id }) => {
        if (!mountedRef.current) return;
        setDeviceId(device_id);
        setPhase("ready");
      });

      player.addListener("not_ready", ({ device_id }) => {
        // SDK marks the device as not_ready when the tab is backgrounded
        // for long enough that Spotify garbage-collects the connection.
        // The SDK reconnects on its own; we just surface that to the UI.
        if (!mountedRef.current) return;
        if (deviceId === device_id) setDeviceId(null);
      });

      player.addListener("initialization_error", ({ message }) => {
        if (!mountedRef.current) return;
        setErrorMessage("SDK init error: " + message);
        setPhase("error");
      });
      player.addListener("authentication_error", ({ message }) => {
        if (!mountedRef.current) return;
        // Token is bad or revoked — treat as needs-relogin so the UI
        // shows the right call-to-action.
        setReloginRequired(true);
        setErrorMessage("Spotify auth error: " + message);
        setPhase("error");
      });
      player.addListener("account_error", ({ message }) => {
        // This is the canonical signal for a non-Premium account.
        if (!mountedRef.current) return;
        setErrorMessage(message || "Spotify Premium is required for Web Playback.");
        setPhase("not_premium");
      });
      player.addListener("playback_error", ({ message }) => {
        if (!mountedRef.current) return;
        setErrorMessage("Playback error: " + message);
      });

      player.addListener("player_state_changed", (state) => {
        if (!mountedRef.current || !state) return;
        setPlayerState(state);
        lastStateAtRef.current = Date.now();
        lastPositionRef.current = state.position || 0;
        setLocalPositionMs(state.position || 0);
      });

      playerRef.current = player;
      try {
        const connected = await player.connect();
        if (!connected && mountedRef.current) {
          setErrorMessage("Spotify SDK refused to connect.");
          setPhase("error");
        }
      } catch (err) {
        if (mountedRef.current) {
          setErrorMessage("Spotify SDK connect threw: " + (err && err.message ? err.message : "unknown"));
          setPhase("error");
        }
      }
    }, []);

    useEffect(() => {
      mountedRef.current = true;
      init();
      return () => {
        mountedRef.current = false;
        const p = playerRef.current;
        if (p && typeof p.disconnect === "function") {
          // Tear the SDK connection down on unmount so we don't leave
          // a zombie 'Hermes Dashboard' device hanging in the user's
          // Spotify Connect list.
          try { p.disconnect(); } catch (_) { /* ignore */ }
        }
        playerRef.current = null;
      };
    }, [init]);

    // ---- Local progress ticking (between SDK state updates) ----------------

    useEffect(() => {
      if (phase !== "ready" || !playerState || playerState.paused) return;
      const id = setInterval(() => {
        if (!mountedRef.current) return;
        const elapsed = Date.now() - lastStateAtRef.current;
        setLocalPositionMs(lastPositionRef.current + elapsed);
      }, 500);
      return () => clearInterval(id);
    }, [phase, playerState]);

    // ---- Transport controls -------------------------------------------------

    const togglePlay = useCallback(() => {
      const p = playerRef.current;
      if (p) p.togglePlay().catch(() => undefined);
    }, []);
    const nextTrack = useCallback(() => {
      const p = playerRef.current;
      if (p) p.nextTrack().catch(() => undefined);
    }, []);
    const prevTrack = useCallback(() => {
      const p = playerRef.current;
      if (p) p.previousTrack().catch(() => undefined);
    }, []);
    const onSeek = useCallback((e) => {
      const p = playerRef.current;
      const ms = Number(e.target.value);
      if (p && Number.isFinite(ms)) {
        lastPositionRef.current = ms;
        lastStateAtRef.current = Date.now();
        setLocalPositionMs(ms);
        p.seek(ms).catch(() => undefined);
      }
    }, []);
    const onVolume = useCallback((e) => {
      const p = playerRef.current;
      const v = Number(e.target.value);
      if (p && Number.isFinite(v)) {
        setVolumeState(v);
        p.setVolume(v).catch(() => undefined);
      }
    }, []);
    const copyDeviceId = useCallback(() => {
      if (!deviceId) return;
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(deviceId).catch(() => undefined);
      }
    }, [deviceId]);

    // ---- Render -------------------------------------------------------------

    const title = h(C.CardTitle, null, "Spotify Web Playback");

    if (phase === "init" || phase === "loading_sdk") {
      return h(C.Card, null,
        h(C.CardHeader, null, title),
        h(C.CardContent, null,
          h("p", { className: "text-sm text-muted-foreground" },
            phase === "init" ? "Checking Spotify login…" : "Loading Spotify Web Playback SDK…"
          )
        )
      );
    }

    if (phase === "logged_out") {
      return h(C.Card, null,
        h(C.CardHeader, null, title),
        h(C.CardContent, { className: "space-y-3" },
          h("p", { className: "text-sm" },
            "Spotify isn't connected yet. Run ",
            h("code", null, "hermes auth spotify"),
            " in a terminal, then reload this tab."
          ),
          h(C.Button, { variant: "outline", onClick: init }, "Retry")
        )
      );
    }

    if (phase === "not_premium") {
      return h(C.Card, null,
        h(C.CardHeader, null, title),
        h(C.CardContent, { className: "space-y-3" },
          h("p", { className: "text-sm" },
            "Spotify Premium is required for in-browser playback. ",
            "You can still control existing Spotify clients (phone, desktop, Sonos) via the ",
            h("code", null, "spotify_playback"),
            " agent tools."
          ),
          errorMessage ? h("p", { className: "text-xs text-muted-foreground" }, errorMessage) : null
        )
      );
    }

    if (phase === "error") {
      return h(C.Card, null,
        h(C.CardHeader, null, title),
        h(C.CardContent, { className: "space-y-3" },
          h("p", { className: "text-sm text-destructive" }, errorMessage || "Something went wrong."),
          reloginRequired
            ? h("p", { className: "text-xs" },
                "Run ",
                h("code", null, "hermes auth spotify"),
                " to re-authenticate, then reload."
              )
            : null,
          h(C.Button, { variant: "outline", onClick: init }, "Retry")
        )
      );
    }

    // phase === "ready"
    const isPlaying = playerState && !playerState.paused;
    const durationMs = (playerState && playerState.duration) || 0;
    const positionMs = Math.min(localPositionMs, durationMs || localPositionMs);
    const name = trackName(playerState) || "—";
    const artists = trackArtists(playerState) || "—";
    const art = trackArtUrl(playerState);

    return h(C.Card, null,
      h(C.CardHeader, { className: "flex flex-row items-center justify-between gap-2" },
        title,
        deviceId ? h(C.Badge, { variant: "secondary" }, DEVICE_NAME + " · ready") : h(C.Badge, null, "registering…")
      ),
      h(C.CardContent, { className: "space-y-4" },
        h("div", { className: "flex items-center gap-4" },
          art
            ? h("img", {
                src: art,
                alt: "Album art",
                className: "h-20 w-20 rounded shadow",
              })
            : h("div", {
                className: "h-20 w-20 rounded bg-muted flex items-center justify-center text-xs text-muted-foreground",
              }, "no art"),
          h("div", { className: "flex-1 min-w-0" },
            h("div", { className: "truncate text-base font-medium" }, name),
            h("div", { className: "truncate text-sm text-muted-foreground" }, artists),
            h("div", { className: "mt-2 flex items-center gap-2 text-xs text-muted-foreground" },
              h("span", null, formatMs(positionMs)),
              h("input", {
                type: "range",
                min: 0,
                max: Math.max(durationMs, 1),
                step: 1000,
                value: positionMs,
                onChange: onSeek,
                disabled: !durationMs,
                className: "flex-1",
                "aria-label": "Seek",
              }),
              h("span", null, formatMs(durationMs))
            )
          )
        ),
        h("div", { className: "flex items-center gap-2" },
          h(C.Button, { variant: "outline", size: "sm", onClick: prevTrack, "aria-label": "Previous" }, "⏮"),
          h(C.Button, {
            variant: isPlaying ? "secondary" : "default",
            size: "sm",
            onClick: togglePlay,
            "aria-label": isPlaying ? "Pause" : "Play",
          }, isPlaying ? "⏸ Pause" : "▶ Play"),
          h(C.Button, { variant: "outline", size: "sm", onClick: nextTrack, "aria-label": "Next" }, "⏭"),
          h("div", { className: "ml-auto flex items-center gap-2 text-xs text-muted-foreground" },
            h("span", null, "Vol"),
            h("input", {
              type: "range",
              min: 0,
              max: 1,
              step: 0.01,
              value: volume,
              onChange: onVolume,
              className: "w-24",
              "aria-label": "Volume",
            })
          )
        ),
        h("div", { className: "text-xs text-muted-foreground" },
          deviceId
            ? h(React.Fragment, null,
                "Tab registered as ",
                h("code", null, DEVICE_NAME),
                ". Agent tool calls that target ",
                h("code", null, "device_id=" + deviceId),
                " play here. ",
                h("button", {
                  type: "button",
                  onClick: copyDeviceId,
                  className: cn("underline underline-offset-2"),
                }, "Copy device id")
              )
            : "Registering as a Spotify Connect device…"
        )
      )
    );
  }

  window.__HERMES_PLUGINS__.register("spotify", SpotifyPage);
})();
