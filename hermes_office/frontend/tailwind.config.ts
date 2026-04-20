import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0f172a",
        ink2: "#1e293b",
        sky1: "#7dd3fc",
        sky2: "#38bdf8",
        carpet: { rest: "#fde68a", learn: "#bae6fd", talk: "#fbcfe8", work: "#bbf7d0" },
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        display: ['"Plus Jakarta Sans"', "Inter", "ui-sans-serif", "sans-serif"],
      },
      boxShadow: {
        soft: "0 12px 40px rgba(15, 23, 42, 0.08)",
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
      },
    },
  },
  plugins: [],
};

export default config;
