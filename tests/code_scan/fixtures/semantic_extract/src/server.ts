/** Module-level JS/TS docstring. */

import express from "express";

export function greet(name: string): string {
  return `Hello, ${name}!`;
}

export class ServerManager {
  start() {
    console.log("Starting...");
  }
}

export const PORT = 3000;

const app = express();
app.listen(3000, () => console.log("Listening"));
app.get("/users", async (req, res) => {
  res.json({ users: [] });
});

async function fetchUser(id: number) {
  return { id };
}

export default ServerManager;
