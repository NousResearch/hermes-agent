# TypeScript Test Template (Jest)

```typescript
import { ExampleClass } from './example';

describe('ExampleClass', () => {
  let example: ExampleClass;
  let mockDependency: jest.Mocked<Dependency>;

  beforeEach(() => {
    mockDependency = {
      getValue: jest.fn()
    } as jest.Mocked<Dependency>;
    example = new ExampleClass(mockDependency);
  });

  test('method should return expected value', () => {
    // Given
    mockDependency.getValue.mockReturnValue('expected');

    // When
    const result = example.method();

    // Then
    expect(result).toBe('expected');
    expect(mockDependency.getValue).toHaveBeenCalled();
  });
});
```

## Patterns

- Use Jest with TypeScript
- Use jest.Mocked for type safety
- Given-When-Then pattern
- Verify all mock calls
- Use beforeEach for setup
