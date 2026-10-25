# Shared UI Components

Framework: React 19 with TypeScript, Vite, React Router, Tailwind CSS v4, and @nous-research/ui.

## ThemeSwitcher

Path: `web/src/components/ThemeSwitcher.tsx`

Full source:

```tsx
import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Palette, Check, Type } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { ListItem } from "@nous-research/ui/ui/components/list-item";
import { BottomSheet } from "@nous-research/ui/ui/components/bottom-sheet";
import { Typography } from "@nous-research/ui/ui/components/typography/index";
import { useBelowBreakpoint } from "@nous-research/ui/hooks/use-below-breakpoint";
import { BUILTIN_THEMES, THEME_DEFAULT_FONT_ID, useTheme } from "@/themes";
import type { DashboardTheme, FontChoice, ThemeListEntry } from "@/themes";
import { useI18n } from "@/i18n";
import { cn } from "@/lib/utils";

/**
 * Compact theme picker mounted next to the language switcher in the header.
 * Each dropdown row shows a 3-stop swatch (background / midground / warm
 * glow) so users can preview the palette before committing. User-defined
 * themes from `~/.hermes/dashboard-themes/*.yaml` use their API-provided
 * definitions so they show real palette swatches just like built-ins.
 *
 * When placed at the bottom of a container (e.g. the sidebar rail), pass
 * `dropUp` so the menu opens above the trigger instead of clipping below
 * the viewport. On viewports below the `sm` breakpoint, `dropUp` uses a
 * bottom sheet portaled to `document.body` so the picker is not clipped by
 * the sidebar (same idea as a responsive Drawer).
 */
export function ThemeSwitcher({ collapsed = false, dropUp = false }: ThemeSwitcherProps) {
  const { themeName, availableThemes, setTheme, fontId, fontChoices, setFont } = useTheme();
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const narrowViewport = useBelowBreakpoint(640);
  const useMobileSheet = Boolean(dropUp && narrowViewport);

  const close = useCallback(() => setOpen(false), []);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, close]);

  useEffect(() => {
    if (!open || useMobileSheet) return;
    const onMouseDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (wrapperRef.current?.contains(target)) return;
      if (dropdownRef.current?.contains(target)) return;
      close();
    };
    document.addEventListener("mousedown", onMouseDown);
    return () => document.removeEventListener("mousedown", onMouseDown);
  }, [open, close, useMobileSheet]);

  const current = availableThemes.find((th) => th.name === themeName);
  const label = current?.label ?? themeName;
  const sheetTitle = t.theme?.title ?? "Theme";

  return (
    <div ref={wrapperRef} className="relative">
      <Button
        ghost
        size={collapsed ? "icon" : undefined}
        onClick={() => setOpen((o) => !o)}
        className={cn(
          collapsed
            ? "text-text-secondary hover:text-foreground hover:bg-transparent"
            : "px-2 py-1 normal-case tracking-normal font-normal text-xs text-text-secondary hover:text-foreground",
        )}
        title={`${t.theme?.switchTheme ?? "Switch theme"}: ${label}`}
        aria-label={t.theme?.switchTheme ?? "Switch theme"}
        aria-expanded={open}
        aria-haspopup="listbox"
      >
        <span className="inline-flex items-center gap-1.5">
          <Palette className="h-3.5 w-3.5" />

          {!collapsed && (
            <Typography
              className="hidden sm:inline text-display tracking-wide text-xs"
            >
              {label}
            </Typography>
          )}
        </span>
      </Button>

      {useMobileSheet && (
        <BottomSheet
          backdropDismissLabel={t.common.close}
          onClose={close}
          open={open}
          title={sheetTitle}
        >
          <div aria-label={sheetTitle} role="listbox">
            <ThemeSwitcherOptions
              availableThemes={availableThemes}
              close={close}
              setTheme={setTheme}
              themeName={themeName}
            />
            <FontSection
              fontChoices={fontChoices}
              fontId={fontId}
              setFont={setFont}
            />
          </div>
        </BottomSheet>
      )}

      {open && !useMobileSheet && (() => {
        const rect = wrapperRef.current?.getBoundingClientRect();
        const dropdown = (
          <div
            ref={dropdownRef}
            aria-label={sheetTitle}
            className={cn(
              "min-w-[240px] max-h-[70dvh] overflow-y-auto",
              "border border-current/20 bg-background-base/95",
              "shadow-[0_12px_32px_-8px_rgba(0,0,0,0.6)]",
              dropUp ? "fixed z-[100]" : "absolute z-50 right-0 top-full mt-1",
            )}
            role="listbox"
            style={
              dropUp && rect
                ? { bottom: window.innerHeight - rect.top + 4, left: rect.left }
                : undefined
            }
          >
            <div className="border-b border-current/20 px-3 py-2">
              <Typography
                className="text-display text-xs tracking-[0.12em] text-text-tertiary"
              >
                {sheetTitle}
              </Typography>
            </div>

            <ThemeSwitcherOptions
              availableThemes={availableThemes}
              close={close}
              setTheme={setTheme}
              themeName={themeName}
            />
            <FontSection
              fontChoices={fontChoices}
              fontId={fontId}
              setFont={setFont}
            />
          </div>
        );
        return dropUp ? createPortal(dropdown, document.body) : dropdown;
      })()}
    </div>
  );
}

function ThemeSwitcherOptions({
  availableThemes,
  close,
  setTheme,
  themeName,
}: ThemeSwitcherOptionsProps) {
  return (
    <>
      {availableThemes.map((th) => {
        const isActive = th.name === themeName;
        const paletteTheme = BUILTIN_THEMES[th.name] ?? th.definition;

        return (
          <ListItem
            active={isActive}
            aria-selected={isActive}
            className="gap-3"
            key={th.name}
            onClick={() => {
              setTheme(th.name);
              close();
            }}
            role="option"
          >
            {paletteTheme ? (
              <ThemeSwatch theme={paletteTheme} />
            ) : (
              <PlaceholderSwatch />
            )}

            <div className="flex min-w-0 flex-1 flex-col gap-0.5">
              <Typography
                className="truncate text-display text-xs tracking-wide"
              >
                {th.label}
              </Typography>
              {th.description && (
                <Typography className="truncate text-xs tracking-normal text-text-tertiary">
                  {th.description}
                </Typography>
              )}
            </div>

            <Check
              className={cn(
                "h-3 w-3 shrink-0 text-midground",
                isActive ? "opacity-100" : "opacity-0",
              )}
            />
          </ListItem>
        );
      })}
    </>
  );
}

const FONT_CATEGORY_LABEL_KEY: Record<FontChoice["category"], "fontSans" | "fontSerif" | "fontMono"> = {
  sans: "fontSans",
  serif: "fontSerif",
  mono: "fontMono",
};

/** Font-override section rendered below the theme list. Lets the user pick
 *  any catalog font independently of the active theme, or "Theme default"
 *  to clear the override. Each row previews itself in its own font. */
function FontSection({ fontChoices, fontId, setFont }: FontSectionProps) {
  const { t } = useI18n();
  const order: FontChoice["category"][] = ["sans", "serif", "mono"];
  return (
    <>
      <div className="mt-1 border-t border-current/20 px-3 pb-1 pt-2">
        <span className="inline-flex items-center gap-1.5">
          <Type className="h-3 w-3 text-text-tertiary" />
          <Typography
            className="text-display text-xs tracking-[0.12em] text-text-tertiary"
          >
            {t.theme?.fontTitle ?? "Font"}
          </Typography>
        </span>
      </div>

      {/* Theme-default (clears the override). */}
      <ListItem
        active={fontId === THEME_DEFAULT_FONT_ID}
        aria-selected={fontId === THEME_DEFAULT_FONT_ID}
        className="gap-3"
        onClick={() => setFont(THEME_DEFAULT_FONT_ID)}
        role="option"
      >
        <span aria-hidden className="h-4 w-9 shrink-0" />
        <div className="flex min-w-0 flex-1 flex-col gap-0.5">
          <Typography className="truncate text-xs tracking-normal">
            {t.theme?.fontDefault ?? "Theme default"}
          </Typography>
          <Typography className="truncate text-xs tracking-normal text-text-tertiary">
            {t.theme?.fontDefaultHint ?? "Use the active theme's font"}
          </Typography>
        </div>
        <Check
          className={cn(
            "h-3 w-3 shrink-0 text-midground",
            fontId === THEME_DEFAULT_FONT_ID ? "opacity-100" : "opacity-0",
          )}
        />
      </ListItem>

      {order.map((cat) => {
        const fonts = fontChoices.filter((f) => f.category === cat);
        if (fonts.length === 0) return null;
        const catLabel = t.theme?.[FONT_CATEGORY_LABEL_KEY[cat]] ?? cat;
        return (
          <div key={cat}>
            <div className="px-3 pb-0.5 pt-1.5">
              <Typography className="text-[0.65rem] uppercase tracking-[0.1em] text-text-tertiary">
                {catLabel}
              </Typography>
            </div>
            {fonts.map((f) => {
              const isActive = f.id === fontId;
              return (
                <ListItem
                  active={isActive}
                  aria-selected={isActive}
                  className="gap-3"
                  key={f.id}
                  onClick={() => setFont(f.id)}
                  role="option"
                >
                  <span aria-hidden className="h-4 w-9 shrink-0" />
                  <div className="flex min-w-0 flex-1 flex-col">
                    {/* Preview the font in its own stack. */}
                    <span
                      className="truncate text-sm"
                      style={{ fontFamily: f.stack }}
                    >
                      {f.label}
                    </span>
                  </div>
                  <Check
                    className={cn(
                      "h-3 w-3 shrink-0 text-midground",
                      isActive ? "opacity-100" : "opacity-0",
                    )}
                  />
                </ListItem>
              );
            })}
          </div>
        );
      })}
    </>
  );
}

function ThemeSwatch({ theme }: { theme: DashboardTheme }) {
  const [c1, c2, c3] = theme.swatchColors ?? [
    theme.palette.background.hex,
    theme.palette.midground.hex,
    theme.palette.warmGlow,
  ];
  return (
    <div
      aria-hidden
      className="flex h-4 w-9 shrink-0 overflow-hidden border border-current/20"
    >
      <span className="flex-1" style={{ background: c1 }} />
      <span className="flex-1" style={{ background: c2 }} />
      <span className="flex-1" style={{ background: c3 }} />
    </div>
  );
}

function PlaceholderSwatch() {
  return (
    <div
      aria-hidden
      className="h-4 w-9 shrink-0 border border-dashed border-current/20"
    />
  );
}

interface ThemeSwitcherOptionsProps {
  availableThemes: ThemeListEntry[];
  close: () => void;
  setTheme: (name: string) => void;
  themeName: string;
}

interface FontSectionProps {
  fontChoices: FontChoice[];
  fontId: string;
  setFont: (id: string) => void;
}

interface ThemeSwitcherProps {
  collapsed?: boolean;
  dropUp?: boolean;
}

```

## ProfileSwitcher

Path: `web/src/components/ProfileSwitcher.tsx`

Full source:

```tsx
import { useMemo } from "react";
import { Users } from "lucide-react";
import {
  Select,
  SelectOption,
} from "@nous-research/ui/ui/components/select";
import { useProfileScope } from "@/contexts/useProfileScope";
import { useI18n } from "@/i18n";
import { cn } from "@/lib/utils";

/**
 * The machine dashboard's single write-target selector.
 *
 * Rendered in the sidebar above the nav. Every management page (Config,
 * Keys, Skills, MCP, Models) reads/writes the selected profile via the
 * fetchJSON ?profile= injection. Hidden when only one profile exists.
 */
export function ProfileSwitcher({ collapsed }: ProfileSwitcherProps) {
  const { profile, currentProfile, profiles, setProfile } = useProfileScope();
  const { t } = useI18n();

  const currentDashboardLabel = useMemo(
    () =>
      (t.app.currentProfileOption ?? "this dashboard ({name})").replace(
        "{name}",
        currentProfile || "default",
      ),
    [currentProfile, t.app.currentProfileOption],
  );

  if (profiles.length < 2) return null;

  const managed = profile || currentProfile || "default";
  const isOther = !!profile && profile !== currentProfile;
  const managingLabel = t.app.managingProfile ?? "Managing profile";

  return (
    <div
      className={cn(
        "flex items-center gap-2 border-b border-current/10 px-3 py-2",
        collapsed && "lg:justify-center lg:px-0",
      )}
      title={managingLabel}
    >
      <Users
        className={cn(
          "h-3.5 w-3.5 shrink-0",
          isOther ? "text-amber-300" : "text-text-tertiary",
        )}
      />

      <Select
        className={cn(
          "min-w-0 flex-1",
          collapsed && "lg:hidden",
          "[&_button]:h-7 [&_button]:border-border [&_button]:bg-background [&_button]:px-2 [&_button]:text-xs",
          "[&_button]:font-sans [&_button]:normal-case [&_button]:tracking-normal",
          "[&_[role=listbox]>div]:font-sans [&_[role=listbox]>div]:text-xs",
          "[&_[role=listbox]>div]:normal-case [&_[role=listbox]>div]:tracking-normal",
          isOther &&
            "[&_button]:border-amber-500/50 [&_button]:text-amber-300",
        )}
        id="hermes-profile-switcher"
        onValueChange={setProfile}
        value={profile}
      >
        <SelectOption value="">{currentDashboardLabel}</SelectOption>

        {profiles
          .filter((name) => name !== currentProfile)
          .map((name) => (
            <SelectOption key={name} value={name}>
              {name}
            </SelectOption>
          ))}
      </Select>

      {collapsed && <span className="sr-only">{managed}</span>}
    </div>
  );
}

interface ProfileSwitcherProps {
  collapsed?: boolean;
}

```

## SidebarStatusStrip

Path: `web/src/components/SidebarStatusStrip.tsx`

Full source:

```tsx
import { Link } from "react-router";
import type { StatusResponse } from "@/lib/api";
import { cn } from "@/lib/utils";
import { useI18n } from "@/i18n";

/** Gateway + session summary for the System sidebar block (no separate strip chrome). */
export function SidebarStatusStrip({ status }: SidebarStatusStripProps) {
  const { t } = useI18n();

  if (status === null) {
    return (
      <div className="px-5 py-1.5" aria-hidden>
        <div className="h-2 w-[80%] max-w-full animate-pulse rounded-sm bg-midground/10" />
      </div>
    );
  }

  const gw = gatewayLine(status, t);
  const { activeSessionsLabel, gatewayStatusLabel } = t.app;

  return (
    <Link
      to="/sessions"
      title={t.app.statusOverview}
      className={cn(
        "block text-left",
        "px-5 pb-2 pt-0.5",
        "text-text-secondary",
        "transition-colors hover:text-midground",
        "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-midground/40",
        "focus-visible:ring-inset",
      )}
    >
      <div className="flex flex-col gap-1 font-sans text-xs leading-snug tracking-[0.08em]">
        <p className="break-words">
          <span className="text-text-tertiary">{gatewayStatusLabel}</span>{" "}
          <span className={cn("font-medium", gw.tone)}>{gw.label}</span>
        </p>

        <p className="break-words">
          <span className="text-text-tertiary">{activeSessionsLabel}</span>{" "}
          <span className="tabular-nums text-text-secondary">
            {status.active_sessions}
          </span>
        </p>
      </div>
    </Link>
  );
}

export function gatewayLine(
  status: StatusResponse,
  t: ReturnType<typeof useI18n>["t"],
): { label: string; tone: string } {
  const g = t.app.gatewayStrip;
  const byState: Record<string, { label: string; tone: string }> = {
    running: { label: g.running, tone: "text-success" },
    starting: { label: g.starting, tone: "text-warning" },
    startup_failed: { label: g.failed, tone: "text-destructive" },
    stopped: { label: g.stopped, tone: "text-muted-foreground" },
  };
  if (status.gateway_state && byState[status.gateway_state]) {
    return byState[status.gateway_state];
  }
  return status.gateway_running
    ? { label: g.running, tone: "text-success" }
    : { label: g.off, tone: "text-muted-foreground" };
}

interface SidebarStatusStripProps {
  status: StatusResponse | null;
}

```

## SidebarFooter

Path: `web/src/components/SidebarFooter.tsx`

Full source:

```tsx
import { Typography } from "@nous-research/ui/ui/components/typography/index";
import type { StatusResponse } from "@/lib/api";
import { cn } from "@/lib/utils";
import { useI18n } from "@/i18n";

export function SidebarFooter({ status }: SidebarFooterProps) {
  const { t } = useI18n();

  return (
    <div
      className={cn(
        "flex shrink-0 items-center justify-between gap-2",
        "px-5 py-2.5",
        "border-t border-current/10",
      )}
    >
      <Typography
        className="font-mono-ui text-xs tabular-nums tracking-[0.08em] text-text-tertiary lowercase"
      >
        {status?.version != null ? `v${status.version}` : "—"}
      </Typography>

      <a
        href="https://nousresearch.com"
        target="_blank"
        rel="noopener noreferrer"
        className={cn(
          "font-sans text-display text-xs tracking-[0.12em] text-midground",
          "transition-opacity hover:opacity-90",
          "focus-visible:rounded-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-midground/40",
        )}
      >
        {t.app.footer.org}
      </a>
    </div>
  );
}

interface SidebarFooterProps {
  status: StatusResponse | null;
}

```

