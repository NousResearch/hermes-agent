(function () {
  "use strict";
  // Aba "Nyx" do dashboard: monta a cena 3D (nyx.js, módulo ES carregado sob demanda) e liga o fluxo
  // de eventos do Hermes (/api/plugins/nyx/eventos, SSE lido via authedFetch pra levar a auth do host).
  const SDK = window.__HERMES_PLUGIN_SDK__;
  if (!SDK || !window.__HERMES_PLUGINS__) return;
  const React = SDK.React;
  const { useEffect, useRef, useState } = SDK.hooks;
  const h = React.createElement;
  // currentScript só existe enquanto este arquivo executa: guarda a base pra achar nyx.js ao lado
  const BASE = new URL("./", document.currentScript ? document.currentScript.src : location.href).href;

  const ROTULO = {
    pensando: "pensando",
    ferramenta: "usando ferramenta",
    ferramenta_fim: "ferramenta terminou",
    falando: "respondendo",
    pronto: "pronto",
    subagente: "delegou a um subagente",
    subagente_fim: "subagente voltou",
  };

  // lê um corpo SSE e entrega cada `data:` já decodificado
  async function lerSSE(resposta, aoEvento) {
    const leitor = resposta.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await leitor.read();
      if (done) return;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const bloco = buf.slice(0, i);
        buf = buf.slice(i + 2);
        for (const linha of bloco.split("\n")) {
          if (!linha.startsWith("data: ")) continue;
          try { aoEvento(JSON.parse(linha.slice(6))); } catch (_) { /* linha quebrada: ignora */ }
        }
      }
    }
  }

  function NyxPage() {
    const palco = useRef(null);
    const cena = useRef(null);
    const [conexao, setConexao] = useState("conectando");
    const [ultimo, setUltimo] = useState(null);
    const [erro, setErro] = useState(null);

    useEffect(() => {
      let vivo = true;
      const abortar = new AbortController();
      import(BASE + "nyx.js")
        .then((m) => { if (vivo) cena.current = m.montar(palco.current); })
        .catch((e) => setErro(String(e && e.message || e)));

      (async function ouvir() {
        while (vivo) {
          try {
            const r = await SDK.authedFetch("/api/plugins/nyx/eventos", {
              signal: abortar.signal, headers: { Accept: "text/event-stream" },
            });
            if (!r.ok) throw new Error("HTTP " + r.status);
            setConexao("ao vivo");
            await lerSSE(r, (ev) => {
              if (cena.current) cena.current.evento(ev);
              setUltimo(ev);
            });
          } catch (_) {
            if (!vivo) return;
          }
          setConexao("reconectando");
          await new Promise((ok) => setTimeout(ok, 1500));
        }
      })();

      return () => {
        vivo = false;
        abortar.abort();
        if (cena.current) cena.current.destruir();
        cena.current = null;
      };
    }, []);

    const ensaiar = () => { SDK.authedFetch("/api/plugins/nyx/ensaio", { method: "POST" }).catch(() => {}); };
    const texto = ultimo
      ? (ROTULO[ultimo.tipo] || ultimo.tipo) + (ultimo.ferramenta ? " · " + ultimo.ferramenta : "")
      : "em repouso";

    return h("div", { className: "nyx-raiz" },
      h("div", { className: "nyx-palco", ref: palco }),
      h("div", { className: "nyx-barra" },
        h("span", { className: "nyx-ponto nyx-" + (conexao === "ao vivo" ? "on" : "off") }),
        h("span", null, "Nyx · " + conexao),
        h("span", { className: "nyx-estado" }, texto),
        h("button", { className: "nyx-botao", onClick: ensaiar, title: "Manda pelo barramento uma sequência de exemplo" }, "ensaio")),
      erro && h("div", { className: "nyx-erro" }, "Não consegui carregar a cena 3D: " + erro));
  }

  window.__HERMES_PLUGINS__.register("nyx", NyxPage);
})();
