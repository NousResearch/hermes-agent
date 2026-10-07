# Jarvis — avatar de partículas

Avatar do Jarvis em WebGL: a imagem de referência é separada em camadas (fundo, corpo e
~220 mil partículas com a cor exata de cada pixel). Em repouso a cena reproduz a foto; mouse,
ondas, voz e estados (ouvindo / pensando / falando) deslocam partículas e corpo pela mesma
função na GPU, então tudo se move junto.

O giro da cabeça é feito com **vistas desenhadas** (`assets/img/vistas/`: ~16°, ~41°, ~48° e perfil de 90°,
geradas no mesmo estilo e enquadramento da foto). A cabeça é um sólido de fatias elípticas medido na frente
(alfa do corpo) e no perfil (vista de 90°). Cada vista vira um conjunto de partículas preso à superfície no
ângulo dela: nesse ângulo, o conjunto é a imagem exata; entre duas vistas, os dois conjuntos giram juntos e se
fundem; o lado esquerdo é o espelho. Dos ombros pra baixo nada gira. A malha escura do sólido tampa o fundo
e o que fica atrás. Os ângulos e o limiar de brilho ficam em `HEAD3D` (`src/config.js`). As vistas de cima e
de baixo (`cima15`, `baixo15`) estão guardadas pro aceno, ainda não usadas.

## Rodar

Os módulos ES e as imagens soltas precisam de um servidor http (em `file://` o navegador bloqueia):

```bash
cd apps/jarvis-avatar
python3 -m http.server 8000      # abra http://localhost:8000
```

Microfone só funciona em `https://` ou `http://localhost`.

## Arquivo único (artifact / duplo clique)

```bash
python3 build.py                 # gera dist/jarvis.html (~5,4 MB, tudo embutido)
```

`dist/` não é versionado: é gerado a partir do código.

## Estrutura

| Caminho | O que tem |
|---|---|
| `index.html` | marcação da interface (status, dock, controles) |
| `src/styles.css` | visual + `@font-face` da Sora |
| `src/config.js` | `CFG` (ajustes), geometria da foto (`IMG`), lista de camadas, parâmetros de URL |
| `src/shaders.js` | GLSL: campo do mouse/ondas, pose, giro 3D (`place3D`), fundo, malha e partículas |
| `src/audio.js` | microfone, `<audio>` externo, voz do navegador → volume + 8 bandas |
| `src/pose.js` | balanço, olhadas, seguir o mouse, reação aos estados (molas amortecidas) |
| `src/model3d.js` | sólido da cabeça (frente + perfil), conjuntos de partículas por vista e malha de profundidade |
| `src/shapes.js` | formas: cabeça, esfera, galáxia, texto |
| `src/main.js` | WebGL, buffers, render (malha + partículas com profundidade), entrada, API pública, loop |
| `assets/img/` | camadas da foto (1672×941): `fundo`, `mascaras`, `corpo-rgb`/`corpo-alfa` (a luz da figura) e `emissao`/`info` (as partículas); `vistas/` com a cabeça girada |
| `assets/fonts/` | Sora 300/600/700 (SIL Open Font License 1.1) |
| `build.py` | empacota tudo em `dist/jarvis.html` (só biblioteca padrão) |

## Integração com o pipeline de voz

`window.JARVIS` existe depois que as camadas carregam; espere o evento `jarvis:ready`:

```js
addEventListener('jarvis:ready', ({ detail: jarvis }) => {
  jarvis.setState('thinking');           // 'idle' | 'listening' | 'thinking' | 'speaking' (null volta ao automático)
  jarvis.attachAudio(audioDoTTS);        // a boca e as partículas seguem o áudio real
});
```

| Chamada | Efeito |
|---|---|
| `setState(s)` | força um estado; qualquer outro valor devolve o controle ao automático |
| `attachAudio(el)` | analisa um `<audio>`/`<video>`; áudio de outra origem precisa de CORS, senão chega silêncio |
| `setLevel(0..1)` | volume manual por frame (quando o áudio não passa pelo navegador) |
| `speak(texto)` | fala com a voz do navegador (envelope de sílabas simulado) |
| `think(ms)` | estado "pensando" por `ms` (padrão 4500) |
| `shape(nome)` | `'head'`, `'sphere'`, `'galaxy'` ou `'text'` (usa o texto do campo) |
| `wave(x, y, força)` | onda a partir de um ponto, em px da imagem |
| `particles`, `time` | leitura: nº de partículas e segundos desde o início |

## Parâmetros de URL e atalhos

- `?estatico` — sem animação ambiente (comparar com a foto) · `?sofundo` — só o cenário, sem o humanoide
- `?fps` — mostra FPS · `?q=0.4..1` — resolução fixa (sem o ajuste automático para GPU fraca)
- `?giro=0.6` ou `?giro=0.6,0.1` — fixa o giro (e o aceno) em radianos: bancada para ajustar a cabeça 3D
- Teclas: **H** esconde a interface · **F** tela cheia · **M** microfone · **P** pensar · **espaço** fala ·
  **1–4** formas · **D** FPS
