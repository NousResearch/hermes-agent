/**
 * Dashboard Plugin SDK + Registry (desktop port)
 *
 * Exposes React, UI components, hooks, and utilities on the window so that
 * plugin bundles can use them without bundling their own copies.
 *
 * Plugins call window.__HERMES_PLUGINS__.register(name, Component) to register
 * their tab component.
 *
 * Ported from web/src/plugins/registry.ts. The registry half is VERBATIM; the
 * only adaptations are the backend-touch helpers (sourced from ./sdk-backend
 * instead of the web's same-origin @/lib/api) and the `components` map (sourced
 * from the desktop UI kit + ./sdk-components shims instead of @nous-research/ui).
 */

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useI18n } from "@/i18n";
import { cn } from "@/lib/utils";

import {
  api,
  authedFetch,
  buildWsAuthParam,
  buildWsUrl,
  fetchJSON,
} from "./sdk-backend";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Label,
  Select,
  SelectOption,
} from "./sdk-components";
import { PluginSlot, registerSlot } from "./slots";

// ---------------------------------------------------------------------------
// timeAgo / isoTimeAgo — ported from web/src/lib/utils.ts (the desktop's
// @/lib/utils only exports `cn`). The plugin SDK contract exposes these.
// ---------------------------------------------------------------------------

/** Relative time from a Unix epoch timestamp (seconds). */
function timeAgo(ts: number): string {
  const delta = Date.now() / 1000 - ts;

  if (delta < 60) {return "just now";}

  if (delta < 3600) {return `${Math.floor(delta / 60)}m ago`;}

  if (delta < 86400) {return `${Math.floor(delta / 3600)}h ago`;}

  if (delta < 172800) {return "yesterday";}

  return `${Math.floor(delta / 86400)}d ago`;
}

/** Relative time from an ISO-8601 timestamp string. */
function isoTimeAgo(iso: string): string {
  const delta = (Date.now() - new Date(iso).getTime()) / 1000;

  if (delta < 0 || Number.isNaN(delta)) {return "unknown";}

  if (delta < 60) {return "just now";}

  if (delta < 3600) {return `${Math.floor(delta / 60)}m ago`;}

  if (delta < 86400) {return `${Math.floor(delta / 3600)}h ago`;}

  return `${Math.floor(delta / 86400)}d ago`;
}

// ---------------------------------------------------------------------------
// Plugin registry — plugins call register() to add their component.
// ---------------------------------------------------------------------------

type RegistryListener = () => void;

const _registered: Map<string, React.ComponentType> = new Map();
const _loadErrors: Map<string, string> = new Map();
const _listeners: Set<RegistryListener> = new Set();

function _notify() {
  for (const fn of _listeners) {
    try { fn(); } catch { /* ignore */ }
  }
}

/** Re-run registry subscribers (e.g. after a plugin script onload, or dev HMR re-inject). */
export function notifyPluginRegistry() {
  _notify();
}

/** Register a plugin component. Called by plugin JS bundles. */
function registerPlugin(name: string, component: React.ComponentType) {
  _loadErrors.delete(name);
  _registered.set(name, component);
  _notify();
}

/** Get a registered component by plugin name. */
export function getPluginComponent(name: string): React.ComponentType | undefined {
  return _registered.get(name);
}

export function getPluginLoadError(name: string): string | undefined {
  return _loadErrors.get(name);
}

export function setPluginLoadError(name: string, message: string) {
  _loadErrors.set(name, message);
  _notify();
}

/** Subscribe to registry changes (returns unsubscribe fn). */
export function onPluginRegistered(fn: RegistryListener): () => void {
  _listeners.add(fn);

  return () => _listeners.delete(fn);
}

/** Get current count of registered plugins. */
export function getRegisteredCount(): number {
  return _registered.size;
}

// ---------------------------------------------------------------------------
// Expose SDK + registry on window
// ---------------------------------------------------------------------------

/**
 * Version of the plugin SDK contract (see ``plugins/sdk.d.ts``). Bump the
 * major on any backwards-incompatible change to the exposed surface;
 * additive changes (new optional fields / helpers) don't require a bump.
 * Exposed at runtime as ``window.__HERMES_PLUGIN_SDK__.sdkVersion`` so a
 * plugin (or a future host-side compatibility gate) can read it.
 */
export const SDK_CONTRACT_VERSION = "1.1.0";

// Window globals for the plugin SDK are declared in ``plugins/sdk.d.ts`` —
// the single source of truth for the public contract. Don't redeclare them
// here (duplicate ambient declarations with differing modifiers conflict).

export function exposePluginSDK() {
  window.__HERMES_PLUGINS__ = {
    register: registerPlugin,
    registerSlot,
  };

  window.__HERMES_PLUGIN_SDK__ = {
    // Contract version of the plugin SDK surface (see plugins/sdk.d.ts).
    // Bump on backwards-incompatible changes; additive changes don't need it.
    sdkVersion: SDK_CONTRACT_VERSION,
    // React core — plugins use these instead of importing react
    React,
    hooks: {
      useState,
      useEffect,
      useCallback,
      useMemo,
      useRef,
      useContext,
      createContext,
    },

    // Hermes API client (desktop-backed: routes through window.hermesDesktop)
    api,
    // Raw fetchJSON for plugin-specific JSON endpoints
    fetchJSON,
    // Authenticated fetch for non-JSON endpoints (uploads / blob downloads).
    // Handles loopback-token vs gated-cookie auth so plugins never read
    // window.__HERMES_SESSION_TOKEN__ directly.
    authedFetch,
    // Build a ws(s):// URL with the correct auth param for the active mode
    // (token in loopback). Use this for any plugin WebSocket instead of
    // hand-assembling the URL.
    buildWsUrl,
    // Lower-level: resolve just the [authParamName, authParamValue] pair, for
    // plugins that need to build the WS URL themselves.
    buildWsAuthParam,

    // UI components — desktop UI kit where available, local shims elsewhere.
    // Keys match the web SDK contract so the same plugin bundles render.
    components: {
      Card,
      CardHeader,
      CardTitle,
      CardContent,
      Badge,
      Button,
      Checkbox,
      Input,
      Label,
      Select,
      SelectOption,
      Separator,
      Tabs,
      TabsList,
      TabsTrigger,
      PluginSlot,
    },

    // Utilities
    utils: { cn, timeAgo, isoTimeAgo },

    // Hooks
    useI18n,
  };
}
