import * as React from "react";
import { load, type Loaded } from "@/lib/api";

/** One panel's data, refreshed on an interval, each panel independent of the others.
 *  Independent on purpose: one 403 (a viewer reading an admin route) must blank its own
 *  card and nothing else. */
export function usePanel<T>(path: string, refreshMs = 15000, nonce = 0): Loaded<T> {
  // `nonce` lets a caller re-read immediately after a write instead of waiting out the
  // poll interval. Appended to the effect deps, so existing call sites are unaffected.
  const [state, setState] = React.useState<Loaded<T>>({ state: "loading" });

  React.useEffect(() => {
    let alive = true;
    const tick = async () => {
      const next = await load<T>(path);
      if (alive) setState(next);
    };
    void tick();
    // A non-positive interval means "read once, then only on a nonce bump". Guarded here
    // rather than at each call site: setInterval(fn, 0) is a tight loop that pins a core
    // and floods the control plane, and it is one typo away at every usePanel call.
    const id = refreshMs > 0 ? window.setInterval(tick, refreshMs) : 0;
    return () => {
      alive = false;
      if (id) window.clearInterval(id);
    };
  }, [path, refreshMs, nonce]);

  return state;
}

export function useReducedMotion() {
  const [reduced, setReduced] = React.useState(
    () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false,
  );
  React.useEffect(() => {
    const query = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    if (!query) return;
    const onChange = () => setReduced(query.matches);
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);
  return reduced;
}

/** Counts to a target. Animates only when the value CHANGES, so a metric that is
 *  steady stays still — constant motion on a dashboard reads as instability. */
export function useCountUp(target: number, durationMs = 650) {
  const reduced = useReducedMotion();
  const [value, setValue] = React.useState(target);
  const previous = React.useRef(target);

  React.useEffect(() => {
    if (reduced || !Number.isFinite(target) || previous.current === target) {
      previous.current = target;
      setValue(target);
      return;
    }
    const from = previous.current;
    previous.current = target;
    let frame = 0;
    const started = performance.now();
    const step = (now: number) => {
      const progress = Math.min(1, (now - started) / durationMs);
      const eased = 1 - Math.pow(1 - progress, 3);
      setValue(Math.round(from + (target - from) * eased));
      if (progress < 1) frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [target, durationMs, reduced]);

  return value;
}

export function useTheme() {
  // An explicit choice always wins. Absent one, follow the operating system: an operator
  // who runs their machine in light mode should not be handed a dark dashboard and have
  // to go find the toggle. Dark remains the fallback when neither is available.
  const [chosen, setChosen] = React.useState<"dark" | "light" | null>(() => {
    try {
      const stored = localStorage.getItem("nova-theme");
      return stored === "dark" || stored === "light" ? stored : null;
    } catch {
      return null;
    }
  });
  const [system, setSystem] = React.useState<"dark" | "light">(() => {
    try {
      return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
    } catch {
      return "dark";
    }
  });

  // Keep following the system while no explicit choice has been made, so a machine that
  // switches at sunset switches the dashboard with it.
  React.useEffect(() => {
    let mq: MediaQueryList;
    try {
      mq = window.matchMedia("(prefers-color-scheme: light)");
    } catch {
      return;
    }
    const onChange = (e: MediaQueryListEvent) => setSystem(e.matches ? "light" : "dark");
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  const theme = chosen ?? system;

  React.useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
  }, [theme]);

  const toggle = React.useCallback(() => {
    setChosen((current) => {
      const next = (current ?? system) === "dark" ? "light" : "dark";
      try {
        // Only an explicit toggle is persisted. Writing the system-derived value on mount
        // would freeze the first observed preference forever.
        localStorage.setItem("nova-theme", next);
      } catch {
        /* storage blocked: the toggle still works, it just does not persist */
      }
      return next;
    });
  }, [system]);

  return { theme, toggle };
}


/** Screen routing through the hash, so a deep link survives a refresh and the browser's
 *  back button works — without adding a router to a single-page control plane. */
export function useRoute(): [string, (next: string) => void] {
  const read = () => window.location.hash.replace(/^#\/?/, "") || "overview";
  const [route, setRoute] = React.useState(read);

  React.useEffect(() => {
    const onHash = () => setRoute(read());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  return [route, (next: string) => { window.location.hash = `/${next}`; }];
}
