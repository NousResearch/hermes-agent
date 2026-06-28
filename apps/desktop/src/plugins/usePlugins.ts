/**
 * usePlugins hook — discovers and loads dashboard plugins (desktop port).
 *
 * 1. Fetches plugin manifests from GET /api/dashboard/plugins
 * 2. Injects CSS <link> tags for plugins that declare css
 * 3. Loads plugin JS bundles via <script> tags
 * 4. Waits for plugins to call register() and resolves them
 *
 * Ported from web/src/plugins/usePlugins.ts. The only adaptations: manifests are
 * fetched through `window.hermesDesktop.api` (cross-origin IPC, not same-origin
 * fetch), and asset URLs are built off the backend's ABSOLUTE origin
 * (`connection.baseUrl`) rather than the web's same-origin `HERMES_BASE_PATH`.
 * The renderer runs from file:// / 127.0.0.1:5174, so plugin <script>/<link>
 * must point at the separate backend origin. Everything else (DEV cache-bust,
 * SRI integrity, onerror/onload register-check, 2s safety timeout) is verbatim.
 */

import { useEffect, useRef, useState } from "react";

import {
  getPluginComponent,
  notifyPluginRegistry,
  onPluginRegistered,
  setPluginLoadError,
} from "./registry";
import { api, getBackendBaseUrl } from "./sdk-backend";
import type { PluginManifest, RegisteredPlugin } from "./types";

export function usePlugins() {
  const [manifests, setManifests] = useState<PluginManifest[]>([]);
  const [plugins, setPlugins] = useState<RegisteredPlugin[]>([]);
  const [loading, setLoading] = useState(true);
  const loadedScripts = useRef<Set<string>>(new Set());

  // Fetch manifests on mount.
  useEffect(() => {
    (api.getPlugins() as Promise<PluginManifest[]>)
      .then((list) => {
        setManifests(list);

        if (list.length === 0) {setLoading(false);}
      })
      .catch(() => setLoading(false));
  }, []);

  // Load plugin assets when manifests arrive.
  useEffect(() => {
    if (manifests.length === 0) {return;}

    let cancelled = false;
    const injectedScripts: HTMLScriptElement[] = [];

    // Resolve the backend's absolute origin once, then inject assets. The web
    // uses a synchronous same-origin HERMES_BASE_PATH; the desktop must await
    // the connection descriptor (window.hermesDesktop.getConnection), so the
    // injection loop is wrapped in an async resolve.
    void getBackendBaseUrl().then((baseUrl) => {
      if (cancelled) {return;}

      for (const manifest of manifests) {
        // Inject CSS if specified.
        if (manifest.css) {
          const cssUrl = `${baseUrl}/dashboard-plugins/${manifest.name}/${manifest.css}`;

          if (!document.querySelector(`link[href="${cssUrl}"]`)) {
            const link = document.createElement("link");
            link.rel = "stylesheet";
            link.href = cssUrl;
            document.head.appendChild(link);
          }
        }

        // Load JS bundle. In dev, cache-bust so Vite HMR can clear the
        // in-memory registry while the browser would otherwise never
        // re-execute a previously cached <script> URL.
        const bundleUrl = `${baseUrl}/dashboard-plugins/${manifest.name}/${manifest.entry}`;

        const scriptSrc = import.meta.env.DEV
          ? `${bundleUrl}?hermes_dv=${Date.now()}`
          : bundleUrl;

        if (!import.meta.env.DEV) {
          if (loadedScripts.current.has(bundleUrl)) {continue;}
          loadedScripts.current.add(bundleUrl);
        }

        const script = document.createElement("script");
        script.setAttribute("data-hermes-plugin", manifest.name);
        script.src = scriptSrc;
        script.async = true;

        // SRI integrity verification — defense against compromised plugin
        // delivery. Plugin manifests can declare an integrity hash
        // (e.g. "sha384-...") which the browser verifies before executing.
        // Without this, a man-in-the-middle or compromised plugin server
        // can substitute the JS bundle silently. Opt-in: when no integrity
        // is declared in the manifest, behavior is unchanged.
        if (manifest.integrity && typeof manifest.integrity === "string") {
          script.integrity = manifest.integrity;
          script.crossOrigin = "anonymous";
        }

        script.onerror = () => {
          setPluginLoadError(manifest.name, "LOAD_FAILED");
          console.warn(
            `[plugins] Failed to load ${manifest.name} from ${scriptSrc} (open Network tab)`,
          );
        };

        script.onload = () => {
          notifyPluginRegistry();
          queueMicrotask(() => {
            if (getPluginComponent(manifest.name)) {return;}
            setPluginLoadError(manifest.name, "NO_REGISTER");
          });
        };

        document.body.appendChild(script);
        injectedScripts.push(script);
      }
    });

    // Give plugins a moment to load and register, then stop loading state.
    const timeout = setTimeout(() => setLoading(false), 2000);

    return () => {
      cancelled = true;
      clearTimeout(timeout);

      if (import.meta.env.DEV) {
        for (const el of injectedScripts) {
          el.remove();
        }
      }
    };
  }, [manifests]);

  // Listen for plugin registrations and resolve them against manifests.
  useEffect(() => {
    function resolvePlugins() {
      const resolved: RegisteredPlugin[] = [];

      for (const manifest of manifests) {
        const component = getPluginComponent(manifest.name);

        if (component) {
          resolved.push({ manifest, component });
        }
      }

      setPlugins(resolved);

      // If all plugins registered, stop loading early.
      if (resolved.length === manifests.length && manifests.length > 0) {
        setLoading(false);
      }
    }

    resolvePlugins();
    const unsub = onPluginRegistered(resolvePlugins);

    return unsub;
  }, [manifests]);

  return { plugins, manifests, loading };
}
