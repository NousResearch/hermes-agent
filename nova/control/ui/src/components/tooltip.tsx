import * as React from "react";
import * as TooltipPrimitive from "@radix-ui/react-tooltip";
import { cn } from "@/lib/utils";

export const TooltipProvider = TooltipPrimitive.Provider;
export const Tooltip = TooltipPrimitive.Root;
export const TooltipTrigger = TooltipPrimitive.Trigger;

export function TooltipContent({
  className, sideOffset = 8, ...props
}: React.ComponentProps<typeof TooltipPrimitive.Content>) {
  return (
    <TooltipPrimitive.Portal>
      <TooltipPrimitive.Content
        data-slot="tooltip-content"
        sideOffset={sideOffset}
        className={cn(
          "glass-elevated text-ink z-50 w-fit max-w-xs rounded-lg px-3 py-2 text-[12px] leading-relaxed",
          "data-[state=delayed-open]:animate-in data-[state=delayed-open]:fade-in-0 data-[state=delayed-open]:zoom-in-95",
          "data-[state=closed]:animate-out data-[state=closed]:fade-out-0",
          className,
        )}
        {...props}
      />
    </TooltipPrimitive.Portal>
  );
}

/** Hover text on a term that needs explaining. Dotted underline so it is discoverable
 *  without a legend, and a real button under it so it reaches the keyboard too. */
export function Hint({ children, text }: { children: React.ReactNode; text: React.ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          className="decoration-ink-faint/50 hover:decoration-ink-muted cursor-help text-left underline decoration-dotted underline-offset-4 transition-colors"
        >
          {children}
        </button>
      </TooltipTrigger>
      <TooltipContent>{text}</TooltipContent>
    </Tooltip>
  );
}

export function InfoDot({ text, label = "What this means" }: { text: React.ReactNode; label?: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label={label}
          className="text-ink-faint/70 hover:text-ink-muted inline-grid size-3.5 place-items-center rounded-full border border-current text-[9px] leading-none font-semibold transition-colors"
        >
          i
        </button>
      </TooltipTrigger>
      <TooltipContent>{text}</TooltipContent>
    </Tooltip>
  );
}
