import { NavLink, Outlet } from "react-router-dom";
import { useGatewayContext } from "@/gateway/GatewayContext";
import { NAV_ITEMS } from "./nav";

export function AppShell() {
  const { skin } = useGatewayContext();
  const chatNav = NAV_ITEMS.filter((n) => n.group === "chat");
  const manageNav = NAV_ITEMS.filter((n) => n.group === "manage");

  return (
    <div className="ht-app">
      <nav className="ht-rail" aria-label="Primary">
        <div className="ht-rail__brand" title={skin.agentName}>
          <span className="ht-rail__mark">◆</span>
        </div>
        <div className="ht-rail__group">
          {chatNav.map((n) => (
            <RailLink key={n.to} to={n.to} label={n.label} glyph={n.glyph} />
          ))}
        </div>
        <div className="ht-rail__divider" />
        <div className="ht-rail__group">
          {manageNav.map((n) => (
            <RailLink key={n.to} to={n.to} label={n.label} glyph={n.glyph} />
          ))}
        </div>
      </nav>
      <div className="ht-content">
        <Outlet />
      </div>
    </div>
  );
}

function RailLink({ to, label, glyph }: { to: string; label: string; glyph: string }) {
  return (
    <NavLink
      to={to}
      end={to === "/"}
      className={({ isActive }) => `ht-rail__link${isActive ? " is-active" : ""}`}
      title={label}
    >
      <span className="ht-rail__glyph" aria-hidden>
        {glyph}
      </span>
      <span className="ht-rail__label">{label}</span>
    </NavLink>
  );
}
