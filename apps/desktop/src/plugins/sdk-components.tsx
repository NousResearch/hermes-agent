/**
 * SDK component shims for primitives the desktop UI kit doesn't ship but plugin
 * bundles consume by key off `window.__HERMES_PLUGIN_SDK__.components`.
 *
 * The web dashboard exposes `@nous-research/ui` Card / Label / Select (a custom
 * combobox that reads `<SelectOption value>` children). The desktop's own UI kit
 * (apps/desktop/src/components/ui) has Badge / Button / Checkbox / Input /
 * Separator / Tabs but NO Card / Label, and its `Select` is a Radix composite
 * with an incompatible API. These thin shims reproduce the web prop contract so
 * the SAME unmodified plugin bundle (e.g. kanban) renders identically.
 */

import {
  Children,
  type ComponentProps,
  isValidElement,
  type ReactElement,
  type ReactNode,
} from "react";

import { cn } from "@/lib/utils";

// ── Card family ────────────────────────────────────────────────────────────
// Mirrors @nous-research/ui Card: a bordered surface that accepts className +
// children. Tones map to the desktop theme's card variables.

export function Card({ className, ...props }: ComponentProps<"div">) {
  return (
    <div
      className={cn(
        "rounded-[4px] border border-(--ui-stroke-secondary) bg-(--ui-bg-secondary) text-(--ui-text-primary)",
        className,
      )}
      data-slot="card"
      {...props}
    />
  );
}

export function CardHeader({ className, ...props }: ComponentProps<"div">) {
  return <div className={cn("flex flex-col gap-1 p-3", className)} data-slot="card-header" {...props} />;
}

export function CardTitle({ className, ...props }: ComponentProps<"div">) {
  return <div className={cn("text-sm font-semibold leading-none", className)} data-slot="card-title" {...props} />;
}

export function CardContent({ className, ...props }: ComponentProps<"div">) {
  return <div className={cn("p-3", className)} data-slot="card-content" {...props} />;
}

// ── Label ──────────────────────────────────────────────────────────────────

export function Label({ className, ...props }: ComponentProps<"label">) {
  return (
    <label
      className={cn("text-xs font-medium leading-none text-(--ui-text-secondary)", className)}
      data-slot="label"
      {...props}
    />
  );
}

// ── Select / SelectOption ──────────────────────────────────────────────────
// Faithful reproduction of the @nous-research/ui contract the plugin bundles
// use: `<Select value onValueChange className>` with `<SelectOption value>`
// children. Implemented as a native <select> so behavior is robust without
// pulling the DS combobox (which the desktop's nous-ui version omits).

interface SelectProps {
  children?: ReactNode;
  className?: string;
  disabled?: boolean;
  id?: string;
  onValueChange?: (value: string) => void;
  placeholder?: string;
  value?: string;
}

export function Select({
  children,
  className,
  disabled,
  id,
  onValueChange,
  value,
}: SelectProps) {
  const options = collectOptions(children);

  return (
    <select
      className={cn(
        "h-9 w-full rounded-[3px] border border-(--ui-stroke-secondary) bg-(--ui-bg-tertiary) px-2 text-sm text-(--ui-text-primary)",
        "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring/40",
        "disabled:cursor-not-allowed disabled:opacity-50",
        className,
      )}
      data-slot="select"
      disabled={disabled}
      id={id}
      onChange={(e) => onValueChange?.(e.target.value)}
      value={value ?? ""}
    >
      {options.map((opt) => (
        <option key={opt.value} value={opt.value}>
          {opt.label}
        </option>
      ))}
    </select>
  );
}

// Marker component — `Select` reads value/children from the tree. Renders
// nothing on its own (matches @nous-research/ui SelectOption).
export function SelectOption(_props: { children?: ReactNode; value: string }) {
  return null;
}

interface SelectOptionData {
  label: string;
  value: string;
}

function collectOptions(children: ReactNode): SelectOptionData[] {
  const out: SelectOptionData[] = [];
  Children.forEach(children, (child) => {
    if (!isValidElement(child)) {return;}
    const el = child as ReactElement<{ children?: ReactNode; value?: unknown }>;

    if (el.props.value !== undefined) {
      out.push({
        label:
          typeof el.props.children === "string"
            ? el.props.children
            : String(el.props.value),
        value: String(el.props.value),
      });
    } else if (el.props.children) {
      out.push(...collectOptions(el.props.children));
    }
  });

  return out;
}
