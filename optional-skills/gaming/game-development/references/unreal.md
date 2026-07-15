# Unreal Engine (C++ / Blueprints) Gameplay Programming

Unreal pairs C++ with **Blueprints** (visual scripting). The idiomatic approach
is **hybrid**: core systems and performance-critical logic in C++, designer-
facing tweaks and rapid iteration in Blueprints derived from those C++ classes.

---

## The Gameplay Framework

Unreal gives you a structured class hierarchy — use it, don't reinvent it:

| Class | Role |
|---|---|
| `AActor` | Anything placeable in a level |
| `APawn` | An Actor that can be possessed/controlled |
| `ACharacter` | A Pawn with a capsule, movement component, mesh (humanoids) |
| `AController` / `APlayerController` | The "brain" possessing a Pawn |
| `AGameModeBase` | Rules, spawning, win/lose (server-authoritative) |
| `AGameStateBase` | Replicated game-wide state |
| `APlayerState` | Per-player state (score, name) |
| `UActorComponent` / `USceneComponent` | Reusable behavior attached to Actors |

**Composition over inheritance:** put reusable behavior in Components
(health, inventory, ability) and attach them, rather than deep class trees.

---

## C++ vs Blueprints — when to use which

| Use C++ for | Use Blueprints for |
|---|---|
| Core systems, base classes | Level scripting, quick prototypes |
| Performance-critical loops | Designer-tweakable values |
| Networking/replication logic | UI (UMG) wiring |
| Anything reused widely | One-off actor behavior |

**Best practice:** write a C++ base class exposing properties/functions via the
reflection macros, then create a Blueprint child for tuning and composition.

---

## The Reflection System (macros)

Unreal's macros expose C++ to the editor, Blueprints, serialization, and
networking. You must use them for anything the engine should "see":

```cpp
// AEnemy.h
#include "Net/UnrealNetwork.h"   // DOREPLIFETIME

UCLASS()
class MYGAME_API AEnemy : public ACharacter
{
    GENERATED_BODY()
public:
    AEnemy();

    // Editable in editor + Blueprint, replicated to clients
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Replicated, Category="Stats")
    float Health = 100.f;

    // Callable from Blueprints
    UFUNCTION(BlueprintCallable, Category="Combat")
    void TakeDamageAmount(float Amount);

protected:
    // Required: without this override a Replicated property never syncs.
    virtual void GetLifetimeReplicatedProps(
        TArray<FLifetimeProperty>& OutProps) const override;
};
```

```cpp
// AEnemy.cpp
#include "AEnemy.h"
#include "Net/UnrealNetwork.h"

AEnemy::AEnemy()
{
    bReplicates = true;   // the actor itself must replicate first
}

void AEnemy::GetLifetimeReplicatedProps(
    TArray<FLifetimeProperty>& OutProps) const
{
    Super::GetLifetimeReplicatedProps(OutProps);
    DOREPLIFETIME(AEnemy, Health);   // register the property for replication
}
```

- `UPROPERTY` — exposes a variable (Inspector, GC tracking, networking).
- `UFUNCTION` — exposes a function (Blueprint-callable, RPCs).
- **`Replicated` alone does nothing.** The actor must set `bReplicates = true`
  and register every replicated property in `GetLifetimeReplicatedProps` with
  `DOREPLIFETIME`, or the value silently never reaches clients.
- **Forgetting `UPROPERTY` on a `UObject*` pointer = it can be garbage
  collected out from under you.** Common beginner crash.

---

## Character + Enhanced Input

Modern Unreal uses the **Enhanced Input** system (Input Actions + Mapping
Contexts), the engine equivalent of the core-systems action map.

```cpp
void AMyCharacter::SetupPlayerInputComponent(UInputComponent* PIC)
{
    Super::SetupPlayerInputComponent(PIC);
    if (auto* EIC = Cast<UEnhancedInputComponent>(PIC))
    {
        EIC->BindAction(MoveAction, ETriggerEvent::Triggered, this, &AMyCharacter::Move);
        EIC->BindAction(JumpAction, ETriggerEvent::Started,   this, &ACharacter::Jump);
    }
}

void AMyCharacter::Move(const FInputActionValue& Value)
{
    const FVector2D Axis = Value.Get<FVector2D>();
    AddMovementInput(GetActorForwardVector(), Axis.Y);
    AddMovementInput(GetActorRightVector(),   Axis.X);
}
```

`ACharacter` already includes `UCharacterMovementComponent` — walking,
jumping, gravity, and networking are handled. Don't rewrite movement from
scratch; configure the component.

---

## Components (reusable behavior)

```cpp
UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class MYGAME_API UHealthComponent : public UActorComponent
{
    GENERATED_BODY()
public:
    UPROPERTY(EditAnywhere, BlueprintReadWrite) float MaxHealth = 100.f;
    UPROPERTY(BlueprintAssignable) FOnDeath OnDeath;  // delegate = event bus

    UFUNCTION(BlueprintCallable)
    void ApplyDamage(float Amount);
};
```

Attach `UHealthComponent` to player, enemies, destructibles — write once, reuse
everywhere. `BlueprintAssignable` delegates are Unreal's event/signal bus.

---

## Blueprints — the idioms

- **Event Graph** — react to events (BeginPlay, Tick, input, overlaps).
- **Functions** — reusable pure/impure logic; keep graphs readable.
- **Avoid Tick when possible** — use timers, events, or overlaps instead of
  per-frame Blueprint logic (Tick in Blueprint is a common perf sink).
- **Blueprint → C++ "nativization" mindset:** prototype in BP, move hot paths
  to C++ when profiling demands it.

---

## Unreal Gotchas

- **Always `UPROPERTY()` your `UObject*` members** or risk GC crashes.
- **GameMode is server-only** — don't put client UI logic there.
- **Compile times in C++ are long** — structure code to minimize header
  dependencies; use forward declarations.
- **Builds are huge** (tens of GB editor, large packaged games) — plan disk and
  build times.
- **Profile with Unreal Insights / `stat` commands** (`stat fps`, `stat unit`,
  `stat game`).
- **Source control:** Unreal assets are binary `.uasset` files — Git LFS is
  mandatory, and coordinate to avoid binary merge conflicts (lock files or
  clear ownership).
