import React from "react";
import { useCounter } from "../hooks/useCounter";

export function App(): React.ReactElement {
  const { count, increment } = useCounter();
  return React.createElement("div", null,
    React.createElement("h1", null, "Counter: " + count),
    React.createElement("button", { onClick: increment }, "Increment")
  );
}
