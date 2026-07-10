import { createContext, useContext, type ReactNode } from "react";
import { useGateway, type UseGateway } from "./useGateway";

// Holds the single live gateway connection above the router so the chat
// session and its streaming state survive navigation to management pages.
const GatewayContext = createContext<UseGateway | null>(null);

export function GatewayProvider({ children }: { children: ReactNode }) {
  const gw = useGateway();
  return <GatewayContext.Provider value={gw}>{children}</GatewayContext.Provider>;
}

export function useGatewayContext(): UseGateway {
  const ctx = useContext(GatewayContext);
  if (!ctx) {
    throw new Error("useGatewayContext must be used within a GatewayProvider");
  }
  return ctx;
}
