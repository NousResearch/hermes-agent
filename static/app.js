const { useState, useEffect } = React;

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [logs, setLogs] = useState([]);
  const [leftWidth, setLeftWidth] = useState(320);
  const [rightWidth, setRightWidth] = useState(280);
  const [dragLeft, setDragLeft] = useState(false);
  const [dragRight, setDragRight] = useState(false);

  // ✅ Logs polling
  useEffect(() => {
    const interval = setInterval(async () => {
      const res = await fetch("/logs");
      const data = await res.json();
      setLogs(data.logs);
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  // ✅ Resize logic
  useEffect(() => {
    function move(e) {
      if (dragLeft) setLeftWidth(Math.max(250, e.clientX));
      if (dragRight) setRightWidth(Math.max(200, window.innerWidth - e.clientX));
    }
    function up() {
      setDragLeft(false);
      setDragRight(false);
    }
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    return () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
  }, [dragLeft, dragRight]);

  // ✅ Streaming chat
  async function sendMessage() {
    if (!input) return;

    setMessages(m => [...m, { role: "user", text: input }]);
    const userInput = input;
    setInput("");

    const res = await fetch("/chat/stream", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ message: userInput })
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    let aiMsg = { role: "assistant", text: "" };
    setMessages(m => [...m, aiMsg]);

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      aiMsg.text += chunk.replace(/data: /g, "");

      setMessages(m => {
        const copy = [...m];
        copy[copy.length - 1] = { ...aiMsg };
        return copy;
      });
    }
  }

  return (
    React.createElement("div", {style:{display:"flex",height:"100vh"}} ,

      React.createElement("div",{style:{width:leftWidth,background:"#f4f4f4",padding:10}},
        React.createElement("h3",null,"Controls"),
        React.createElement("button",{onClick:()=>fetch("/restart",{method:"POST"})},"Restart")
      ),

      React.createElement("div",{style:{width:5,cursor:"col-resize",background:"#ddd"},onMouseDown:()=>setDragLeft(true)}),

      React.createElement("div",{style:{flex:1,display:"flex",flexDirection:"column"}},
        React.createElement("div",{style:{flex:1,overflow:"auto",padding:10}},
          messages.map((m,i)=>
            React.createElement("div",{key:i,style:{textAlign:m.role==="user"?"right":"left"}},
              React.createElement("div",{style:{
                display:"inline-block",
                background:m.role==="user"?"black":"#eee",
                color:m.role==="user"?"white":"black",
                padding:10,
                margin:5,
                borderRadius:10
              }},m.text)
            )
          )
        ),

        React.createElement("div",{style:{display:"flex",borderTop:"1px solid #ccc"}},
          React.createElement("input",{value:input,onChange:e=>setInput(e.target.value),style:{flex:1,padding:10}}),
          React.createElement("button",{onClick:sendMessage},"Send")
        )
      ),

      React.createElement("div",{style:{width:5,cursor:"col-resize",background:"#ddd"},onMouseDown:()=>setDragRight(true)}),

      React.createElement("div",{style:{width:rightWidth,background:"#111",color:"#0f0",padding:10,overflow:"auto"}},
        logs.map((l,i)=>React.createElement("div",{key:i},l))
      )
    )
  );
}

ReactDOM.createRoot(document.getElementById("root"))
  .render(React.createElement(App));
``
