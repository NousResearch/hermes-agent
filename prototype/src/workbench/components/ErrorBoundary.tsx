"use client";

import { Component, type ErrorInfo, type ReactNode } from "react";

/** Keeps one broken view from blanking the whole workbench. */
export class ErrorBoundary extends Component<{ children: ReactNode; label?: string; compact?: boolean }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) { return { error }; }
  componentDidCatch(error: Error, info: ErrorInfo) { console.error(`[aro] ${this.props.label ?? "view"} crashed`, error, info.componentStack); }
  reset = () => this.setState({ error: null });
  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div role="alert" className={`flex flex-1 flex-col items-center justify-center gap-3 text-center ${this.props.compact ? "p-4" : "p-10"}`}>
        <div className="flex size-11 items-center justify-center rounded-[12px] border border-rose/30 bg-rose-tint font-mono text-[15px] font-bold text-rose">!</div>
        <div>
          <p className="font-display text-[14px] font-semibold text-ink">{this.props.label ?? "This panel"} hit an error</p>
          <p className="mt-1 max-w-[360px] font-mono text-[11px] leading-[1.5] text-ink-4">{this.state.error.message}</p>
        </div>
        <div className="flex gap-2">
          <button onClick={this.reset} className="h-[30px] cursor-pointer rounded-[7px] bg-iris px-3 text-[12.5px] font-medium text-on-iris">Try again</button>
          <button onClick={() => navigator.clipboard?.writeText(`${this.state.error?.stack ?? this.state.error?.message}`)} className="h-[30px] cursor-pointer rounded-[7px] border border-line px-3 text-[12.5px] text-ink-2 hover:bg-hover">Copy details</button>
        </div>
      </div>
    );
  }
}
