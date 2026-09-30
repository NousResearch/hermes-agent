/**
 * Standalone host — bundles a REAL React and mounts the Mission Control root.
 *
 * Only the standalone server (`samagent/ui_server.py`) loads this. The Hermes
 * dashboard has its own host and never touches this file; there the SDK already
 * exposes React and the dashboard renders the registered component.
 *
 * Bundling React here (rather than reusing a hand-rolled shim) is what makes
 * hooks, reconciliation and controlled inputs behave. The dashboard path and
 * this path then present the identical `window.__HERMES_PLUGIN_SDK__` surface.
 */
import React from 'react';
import { createRoot } from 'react-dom/client';

const SDK_VERSION = '1.1.0';

const hooks = {
  useState: React.useState,
  useEffect: React.useEffect,
  useCallback: React.useCallback,
  useMemo: React.useMemo,
  useRef: React.useRef,
  useContext: React.useContext,
  createContext: React.createContext,
};

const registered = new Map();
let notify = () => {};

export const pluginRegistry = {
  register(name, component) {
    registered.set(name, component);
    notify();
  },
  registerSlot(name, slot, component) {
    const entry = registered.get(name) || {};
    entry[slot] = component;
    registered.set(name, entry);
  },
  get(name) {
    return registered.get(name);
  },
};

window.__HERMES_PLUGINS__ = pluginRegistry;
window.__HERMES_PLUGIN_SDK__ = {
  sdkVersion: SDK_VERSION,
  React,
  hooks,
  components: {},
  utils: {},
};

const MOUNT_ID = 'samagent-standalone-root';

function StandaloneHost() {
  const [, force] = React.useReducer((n) => n + 1, 0);
  React.useEffect(() => {
    notify = force;
    return () => {
      notify = () => {};
    };
  }, []);

  const Root = registered.get('samagent');
  if (!Root) return null;
  return React.createElement(Root, null);
}

const container = document.getElementById(MOUNT_ID) || document.body;
createRoot(container).render(React.createElement(StandaloneHost));
