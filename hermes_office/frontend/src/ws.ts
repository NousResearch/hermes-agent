// Reconnecting WebSocket subscriber.

import type { ActivityEvent } from "./types";

export type WsMessage =
  | { kind: "hello"; version: string }
  | (ActivityEvent & { kind: ActivityEvent["kind"] });

type Listener = (msg: WsMessage) => void;

export class OfficeSocket {
  private ws: WebSocket | null = null;
  private listeners = new Set<Listener>();
  private retry = 0;
  private alive = true;

  constructor(private url: string = wsUrl()) {
    this.connect();
  }

  private connect() {
    try {
      this.ws = new WebSocket(this.url);
    } catch (e) {
      this.scheduleReconnect();
      return;
    }
    this.ws.addEventListener("open", () => {
      this.retry = 0;
    });
    this.ws.addEventListener("message", (e) => {
      try {
        const msg = JSON.parse(e.data) as WsMessage;
        this.listeners.forEach((l) => l(msg));
      } catch {
        // ignore
      }
    });
    this.ws.addEventListener("close", () => {
      if (this.alive) this.scheduleReconnect();
    });
    this.ws.addEventListener("error", () => {
      this.ws?.close();
    });
  }

  private scheduleReconnect() {
    this.retry = Math.min(this.retry + 1, 6);
    const delay = 250 * Math.pow(2, this.retry);
    setTimeout(() => {
      if (this.alive) this.connect();
    }, delay);
  }

  subscribe(l: Listener): () => void {
    this.listeners.add(l);
    return () => this.listeners.delete(l);
  }

  close() {
    this.alive = false;
    this.ws?.close();
  }
}

function wsUrl(): string {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}/ws/office`;
}
