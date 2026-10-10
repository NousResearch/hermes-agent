# PHP Test Template (PHPUnit)

```php
<?php

use PHPUnit\Framework\TestCase;
use Example\ExampleClass;

class ExampleClassTest extends TestCase
{
    private $mockDependency;
    private $example;

    protected function setUp(): void
    {
        $this->mockDependency = $this->createMock(Dependency::class);
        $this->example = new ExampleClass($this->mockDependency);
    }

    public function testMethodReturnsExpectedValue(): void
    {
        // Given
        $this->mockDependency->method('getValue')
            ->willReturn('expected');

        // When
        $result = $this->example->method();

        // Then
        $this->assertEquals('expected', $result);
        $this->mockDependency->method('getValue')
            ->with()
            ->willReturn('expected');
    }
}
```

## Patterns

- Use PHPUnit
- Use createMock for mocks
- Given-When-Then pattern
- Verify all mock calls
- Use setUp for setup
