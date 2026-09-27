'use strict';
const scenarios = {
  research: { request: 'Czesiek, porównaj trzy narzędzia do newslettera dla małej firmy.', reply: 'Jasne. Jaki masz budżet i ile osób jest na liście? To pomoże wybrać coś, co faktycznie pasuje.', worker: 'Researcher sprawdza informacje i źródła.', result: 'Przykładowy wynik: tabela porównawcza, źródła cen i rekomendacja dopasowana do Twoich kryteriów.' },
  day: { request: 'Mam dziś sporo na głowie. Pomóż mi ustalić, od czego zacząć.', reply: 'Co musi być gotowe dzisiaj, a co może poczekać? Podaj też stałe spotkania. Ułożymy plan, który mieści się w Twoim dniu.', worker: 'Asystent dnia porządkuje priorytety i terminy.', result: 'Przykładowy wynik: trzy priorytety, proponowane bloki pracy i lista spraw do przełożenia.' },
  project: { request: 'Chcę przygotować nową stronę dla mojej firmy. Rozpiszmy pracę.', reply: 'Jaki jest cel strony, dla kogo ją robimy i na kiedy? Na tej podstawie rozdzielimy treści, projekt i wdrożenie.', worker: 'Koordynator dzieli pracę na zadania i zależności.', result: 'Przykładowy wynik: plan etapów, zadania dla agentów oraz lista decyzji, które wymagają Twojej akceptacji.' }
};
const tabs = [...document.querySelectorAll('[data-scenario]')];
const panel = document.querySelector('#scenario');
const demo = document.querySelector('.demo');
const run = document.querySelector('#demo-run');
const steps = [...document.querySelectorAll('#demo-steps li')];
const status = document.querySelector('#demo-status');
const result = document.querySelector('#demo-result p');
let active = 'research';
let timers = [];
function reset() {
  timers.forEach(clearTimeout); timers = [];
  delete demo.dataset.running;
  steps.forEach(step => step.classList.remove('done'));
  status.textContent = 'Gotowy do pokazania';
  result.textContent = 'Wybierz przykład i zobacz, jak cel zmienia się w zadanie.';
  run.disabled = false;
  run.textContent = 'Pokaż przykładowy przebieg →';
}
function select(tab) {
  reset(); active = tab.dataset.scenario;
  tabs.forEach(item => { item.setAttribute('aria-selected', String(item === tab)); item.tabIndex = item === tab ? 0 : -1; });
  panel.setAttribute('aria-labelledby', tab.id);
  document.querySelector('#request').textContent = scenarios[active].request;
  document.querySelector('#reply').textContent = scenarios[active].reply;
  steps[1].querySelector('p').textContent = scenarios[active].worker;
}
tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => select(tab));
  tab.addEventListener('keydown', event => {
    const move = { ArrowRight: (index + 1) % tabs.length, ArrowLeft: (index + tabs.length - 1) % tabs.length, Home: 0, End: tabs.length - 1 };
    if (!(event.key in move)) return;
    event.preventDefault(); const next = tabs[move[event.key]]; select(next); next.focus();
  });
});
run.addEventListener('click', () => {
  reset(); demo.dataset.running = 'true'; run.disabled = true; run.textContent = 'Pokazuję przebieg…';
  const labels = ['Doprecyzowanie celu', 'Delegowanie pracy', 'Przykładowy wynik'];
  const interval = window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 850;
  steps.forEach((step, index) => timers.push(setTimeout(() => {
    step.classList.add('done'); status.textContent = labels[index];
    if (index === steps.length - 1) { result.textContent = scenarios[active].result; run.disabled = false; run.textContent = 'Pokaż jeszcze raz →'; }
  }, interval * index)));
});
