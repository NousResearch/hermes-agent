// Lock screen: shown whenever the API answers 401. The access code is the
// server's --token / HERMES_HUB_TOKEN value; it is stored locally and sent
// as a bearer token on every request.

import { h, clear } from "./utils.js";
import { setToken, getToken } from "./api.js";

let lockEl = null;

export function showLockScreen({ onUnlocked }) {
  if (lockEl) return;

  const input = h("input.input.lock-input", {
    type: "password",
    placeholder: "Access code",
    "aria-label": "Access code",
    autocomplete: "current-password",
    value: "",
  });
  const status = h("div.lock-status.muted.small", {}, "This console is locked.");

  const form = h("form.lock-form", {
    onsubmit: async (ev) => {
      ev.preventDefault();
      const code = input.value.trim();
      if (!code) return;
      setToken(code);
      status.textContent = "VERIFYING…";
      try {
        // Any protected endpoint verifies the code; news is cheap.
        const res = await fetch("/api/state/rev", {
          headers: { Authorization: `Bearer ${code}` },
        });
        if (res.status === 401) throw new Error("bad code");
        status.textContent = "ACCESS GRANTED";
        setTimeout(() => {
          lockEl?.remove();
          lockEl = null;
          onUnlocked?.();
        }, 250);
      } catch {
        setToken(getToken() === code ? "" : getToken());
        status.textContent = "ACCESS DENIED — check the code and try again.";
        input.select();
      }
    },
  },
    input,
    h("button.btn.btn-primary", { type: "submit" }, "Unlock"),
  );

  lockEl = h("div.lock-backdrop", { role: "dialog", "aria-label": "Locked" },
    h("div.lock-panel", {},
      h("div.lock-mark", { "aria-hidden": "true" }, "◆"),
      h("div.lock-title", {}, "HERMES", h("span.brand-sub", {}, "//HUB")),
      h("div.lock-sub", {}, "RESTRICTED — ENTER ACCESS CODE"),
      form,
      status,
      h("p.muted.small.lock-hint", {},
        "The access code is the --token (or HERMES_HUB_TOKEN) the server was started with."),
    ),
  );
  document.body.append(lockEl);
  input.focus();
}
