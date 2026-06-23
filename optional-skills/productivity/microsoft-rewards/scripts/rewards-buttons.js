JSON.stringify(
  Array.from(document.querySelectorAll("button"))
    .map((b, i) => ({
      i,
      text: b.innerText.replace(/\s+/g, " ").trim().substring(0, 60)
    }))
    .filter(t => t.text.length > 5)
)
