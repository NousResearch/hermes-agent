# JavaScript Test Template (Jest)

```javascript
const { ExampleClass } = require('./example');

describe('ExampleClass', () => {
  let example;
  let mockDependency;

  beforeEach(() => {
    mockDependency = {
      getValue: jest.fn()
    };
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

- Use Jest
- Use jest.fn() for mocks
- Given-When-Then pattern
- Verify all mock calls
- Use beforeEach for setup
