# Unity (C#) Gameplay Programming

Idiomatic Unity patterns for gameplay. Unity is component-based: you attach
`MonoBehaviour` scripts to `GameObjects` in a scene. Work *with* this model,
not against it.

---

## MonoBehaviour Lifecycle

The methods Unity calls for you, in order:

| Method | When | Use for |
|---|---|---|
| `Awake()` | On object creation, before Start | Cache references, init self |
| `OnEnable()` | Each time enabled | Subscribe to events |
| `Start()` | Before first frame | Init that depends on other objects |
| `Update()` | Every frame | Input, non-physics movement, timers |
| `FixedUpdate()` | Fixed rate | Physics (Rigidbody forces/velocity) |
| `LateUpdate()` | After all Updates | Camera follow (after target moved) |
| `OnDisable()` | When disabled | Unsubscribe from events |
| `OnDestroy()` | On destruction | Cleanup |

**Cache, don't `GetComponent` every frame:**
```csharp
private Rigidbody2D _rb;
void Awake() { _rb = GetComponent<Rigidbody2D>(); }  // once
void FixedUpdate() { _rb.velocity = ...; }            // reuse
```

---

## Player Controller (2D, physics-based)

```csharp
using UnityEngine;

[RequireComponent(typeof(Rigidbody2D))]
public class PlayerController : MonoBehaviour
{
    [SerializeField] private float moveSpeed = 8f;
    [SerializeField] private float jumpForce = 14f;
    [SerializeField] private LayerMask groundLayer;
    [SerializeField] private Transform groundCheck;

    private Rigidbody2D _rb;
    private float _moveInput;
    private bool _jumpQueued;
    private float _coyoteTimer;       // game feel: grace after leaving ledge
    private const float CoyoteTime = 0.1f;

    void Awake() => _rb = GetComponent<Rigidbody2D>();

    void Update()  // input in Update (every frame), not FixedUpdate
    {
        _moveInput = Input.GetAxisRaw("Horizontal");
        if (Input.GetButtonDown("Jump")) _jumpQueued = true;  // buffer
    }

    void FixedUpdate()  // physics in FixedUpdate
    {
        _rb.velocity = new Vector2(_moveInput * moveSpeed, _rb.velocity.y);

        bool grounded = Physics2D.OverlapCircle(groundCheck.position, 0.15f, groundLayer);
        _coyoteTimer = grounded ? CoyoteTime : _coyoteTimer - Time.fixedDeltaTime;

        if (_jumpQueued && _coyoteTimer > 0f)
        {
            _rb.velocity = new Vector2(_rb.velocity.x, jumpForce);
            _coyoteTimer = 0f;
        }
        _jumpQueued = false;
    }
}
```

Note the discipline: **input in `Update`, physics in `FixedUpdate`**, plus
coyote time and jump buffering for feel.

---

## ScriptableObjects (data-driven design)

Unity's answer to data files. Create reusable data assets in the editor:

```csharp
[CreateAssetMenu(menuName = "Game/EnemyData")]
public class EnemyData : ScriptableObject
{
    public string enemyName;
    public int maxHealth;
    public float moveSpeed;
    public int damage;
    public GameObject prefab;
}
```

Designers create/tune `EnemyData` assets in the Inspector with no code changes.
Also ideal for event channels, config, and shared state.

---

## Prefabs

Reusable GameObject templates. Make anything spawned repeatedly (enemies,
bullets, pickups) a prefab. Instantiate:
```csharp
Instantiate(bulletPrefab, muzzle.position, muzzle.rotation);
```
But for frequently-spawned objects, **pool instead of Instantiate/Destroy**
(see core-systems). Use prefab **variants** for shared-base-with-tweaks.

---

## Enemy AI — FSM

```csharp
public enum AIState { Patrol, Chase, Attack }

public class EnemyAI : MonoBehaviour
{
    [SerializeField] private EnemyData data;
    [SerializeField] private float chaseRange = 6f, attackRange = 1.5f;
    private Transform _player;
    private AIState _state = AIState.Patrol;

    void Start() => _player = GameObject.FindWithTag("Player").transform;

    void Update()
    {
        float dist = Vector2.Distance(transform.position, _player.position);
        _state = dist <= attackRange ? AIState.Attack
               : dist <= chaseRange  ? AIState.Chase
               : AIState.Patrol;

        switch (_state)
        {
            case AIState.Patrol: Patrol(); break;
            case AIState.Chase:  MoveToward(_player.position); break;
            case AIState.Attack: Attack(); break;
        }
    }
    // Patrol/MoveToward/Attack implementations...
}
```

---

## Input System (new)

Prefer the **Input System package** over legacy `Input.GetAxis` for new
projects — it supports rebinding, multiple devices, and action maps natively
(matches the core-systems "actions" pattern). Define an Input Actions asset,
generate a C# class, subscribe to action callbacks.

---

## Coroutines (time-based sequences)

For "do X, wait, do Y" without blocking:
```csharp
IEnumerator FlashRed()
{
    sprite.color = Color.red;
    yield return new WaitForSeconds(0.1f);
    sprite.color = Color.white;
}
StartCoroutine(FlashRed());
```
Great for damage flashes, spawning waves, timed effects. For complex async,
consider UniTask/async-await later.

---

## Unity Gotchas

- **Don't use `GameObject.Find` / `GetComponent` in `Update`** — cache in
  `Awake`/`Start`. They're slow.
- **`Time.deltaTime` for frame movement**, `Time.fixedDeltaTime` in `FixedUpdate`.
- **Null checks on destroyed objects** — Unity overrides `==` so destroyed
  objects compare to null, but stale references still bite.
- **Profile with the Unity Profiler** — watch GC allocations; per-frame `new`
  causes stutter.
- **`[SerializeField] private`** over `public` — exposes to Inspector without
  breaking encapsulation.
