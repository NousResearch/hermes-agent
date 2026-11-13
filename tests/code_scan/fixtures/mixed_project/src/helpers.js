// Helper utilities for mixed project

export function formatDate(date) {
  const d = new Date(date);
  return d.toISOString().split("T")[0];
}

export function slugify(text) {
  return text
    .toLowerCase()
    .replace(/\s+/g, "-")
    .replace(/[^\w-]/g, "");
}

export function debounce(fn, ms) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}
