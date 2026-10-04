import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "GameFi Research Workflow for Hermes Agent",
  description:
    "A structured Hermes workflow for game project discovery, public repository review, and clean research summaries. Showcase demo — static, fictional sample data.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-slate-950 text-slate-100 antialiased">
        {children}
      </body>
    </html>
  );
}
