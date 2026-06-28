import { useSyncExternalStore } from "react";

import { GlyphSpinner } from "@/components/ui/glyph-spinner";
import { useI18n } from "@/i18n";
import { cn } from "@/lib/utils";

import {
  getPluginComponent,
  getPluginLoadError,
  onPluginRegistered,
} from "./registry";

/** Renders a plugin tab once its bundle has called `register()`. */
export function PluginPage({ name }: { name: string }) {
  const { t } = useI18n();

  // Subscribe in render (via useSyncExternalStore) so we never miss
  // `register()` if the script loads before a useEffect would run.
  const Component = useSyncExternalStore(
    (onChange) => onPluginRegistered(onChange),
    () => getPluginComponent(name) ?? null,
    () => null,
  );

  const loadError = useSyncExternalStore(
    (onChange) => onPluginRegistered(onChange),
    () => getPluginLoadError(name) ?? null,
    () => null,
  );

  if (Component) {
    // The desktop pane wraps content in `overflow-hidden` and built-in views
    // bring their own scroll host; plugins (kanban, achievements) render raw,
    // so without this they get clipped and don't fill the pane. Give plugin
    // views a full-size, scrollable host (the web's page layout does this).
    return (
      <div className="h-full w-full min-h-0 overflow-auto">
        <Component />
      </div>
    );
  }

  if (loadError) {
    const message = formatPluginError(loadError);

    return (
      <div
        className={cn("max-w-lg p-4", "text-sm tracking-[0.08em] text-muted-foreground")}
        role="alert"
      >
        {message}
      </div>
    );
  }

  return (
    <div
      className={cn("flex items-center gap-2 p-4", "text-sm tracking-[0.1em] text-muted-foreground")}
    >
      <GlyphSpinner className="shrink-0" />
      <span>{t.common.loading}</span>
    </div>
  );
}

// Desktop i18n's `common` block doesn't carry the dashboard's
// `pluginLoadFailed` / `pluginNotRegistered` keys, so the two error strings are
// inlined here (English) rather than threaded through every locale catalog.
function formatPluginError(code: string): string {
  if (code === "LOAD_FAILED")
    {return "This plugin failed to load. Check the network tab and reload.";}

  if (code === "NO_REGISTER")
    {return "This plugin loaded but did not register a view.";}

  return code;
}
